# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
from collections import deque
import statistics

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv
from rsl_rl.datasets.motion_loader import AMPLoader
from rsl_rl.utils.utils import Normalizer
from rsl_rl.modules.estimator import LSTMEncoder
from copy import copy, deepcopy
import warnings
class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.lstm_encoder_cfg = train_cfg["LSTMEncoder"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        if self.env.include_history_steps is not None:
            num_actor_obs = self.env.num_obs * self.env.include_history_steps
        else:
            num_actor_obs = self.env.num_obs

        self.if_student = self.cfg["if_student"]
        if self.if_student:
            train_cfg['runner']['amp_num_preload_transitions'] = 10

        actor_critic: ActorCritic = actor_critic_class( num_actor_obs=num_actor_obs,
                                                        num_critic_obs=num_critic_obs,
                                                        num_actions=self.env.num_actions,
                                                        **self.policy_cfg).to(self.device)

        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        min_std = (
            torch.tensor(self.cfg["min_normalized_std"], device=self.device) *
            (torch.abs(self.env.dof_pos_limits[:, 1] - self.env.dof_pos_limits[:, 0])))
        
        if self.if_student:
            lstm_encoder = LSTMEncoder(self.lstm_encoder_cfg["input_size"],self.lstm_encoder_cfg["hidden_size"],self.lstm_encoder_cfg["num_layers"]).to(self.device)
            lstm_actor = deepcopy(actor_critic)
            print(lstm_encoder)
        else:
            lstm_encoder = None
            lstm_actor = None 


        self.alg: PPO = alg_class(actor_critic,lstm_encoder, self.lstm_encoder_cfg, lstm_actor, device=self.device, min_std=min_std, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [num_actor_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        self.learn = self.learn if not self.if_student else self.learn_lstm
        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos, reset_env_ids = self.env.step(actions)

                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                    self.alg.process_env_step(rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()

            if(it == 0):
                os.system("cp ./legged_gym/envs/b2w/b2w_config.py " + self.log_dir + "/")
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)


    def learn_lstm(self, num_learning_iterations, init_at_random_ep_len=False):
        print("hellow lstm")
        tot_iter = self.current_learning_iteration + num_learning_iterations
        self.start_learning_iteration = copy(self.current_learning_iteration)
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        critic_obs = self.env.get_privileged_observations()
        obs = self.env.get_observations()

        infos = {}
        # infos["depth"] = self.env.depth_buffer.clone().to(self.device)[:, -1] if self.if_depth else None
        # infos["depth"] = self.env.depth_buffer.clone().to(self.device) if self.if_depth else None
        # infos["delta_yaw_ok"] = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)
        self.alg.lstm_encoder.train()
        self.alg.lstm_actor.train()

        num_pretrain_iter = 0
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            scan_latent_buffer = []
            scandots_latent_buffer = []
            actions_teacher_buffer = []
            actions_student_buffer = []

            for i in range(self.lstm_encoder_cfg["num_steps_per_env"]):
                with torch.no_grad():
                    scandots_latent = self.alg.actor_critic.infer_latent(obs)   
                scandots_latent_buffer.append(scandots_latent)
                obs_prop = obs[:, 29 : 29 + 57].clone() # 本体观测
                scan_latent = self.alg.lstm_encoder(obs_prop)                     
                scan_latent_buffer.append(scan_latent)

                with torch.no_grad():
                    actions_teacher = self.alg.actor_critic.act_inference(obs,scandots_latent=None)
                    actions_teacher_buffer.append(actions_teacher)

                obs_student = obs.clone()
                actions_student = self.alg.lstm_actor.act_inference(obs_student, scandots_latent=scan_latent)
                actions_student_buffer.append(actions_student)

                # detach actions before feeding the env
                if it < num_pretrain_iter:
                    obs, critic_obs, rewards, dones, infos,_= self.env.step(actions_teacher.detach())  # obs has changed to next_obs !! if done obs has been reset
                else:
                    obs, critic_obs, rewards, dones, infos,_= self.env.step(actions_student.detach())  # obs has changed to next_obs !! if done obs has been reset

                # critic_obs = privileged_obs if privileged_obs is not None else obs
                obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)



                if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                
            stop = time.time()
            collection_time = stop - start
            start = stop

            scandots_latent_buffer = torch.cat(scandots_latent_buffer, dim=0)
            scan_latent_buffer = torch.cat(scan_latent_buffer, dim=0)

            actions_teacher_buffer = torch.cat(actions_teacher_buffer, dim=0)
            actions_student_buffer = torch.cat(actions_student_buffer, dim=0)

            #lstm_actor_loss = self.alg.update_depth_actor(actions_student_buffer, actions_teacher_buffer)

            lstm_encoder_loss, lstm_actor_loss = self.alg.update_depth_both(scan_latent_buffer, scandots_latent_buffer, actions_student_buffer, actions_teacher_buffer)
            stop = time.time()
            learn_time = stop - start

            self.alg.lstm_encoder.detach_hidden_states()

            if self.log_dir is not None:
                self.log_lstm(locals())
            if (it-self.start_learning_iteration < 2500 and it % self.save_interval == 0) or \
               (it-self.start_learning_iteration < 5000 and it % (2*self.save_interval) == 0) or \
               (it-self.start_learning_iteration >= 5000 and it % (5*self.save_interval) == 0):
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()


    def log_lstm(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        wandb_dict = {}
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss_lstm/lstm_actor', locs['lstm_actor_loss'], locs['it'])
        self.writer.add_scalar('Loss_lstm/lstm_encoder', locs['lstm_encoder_loss'], locs['it'])

        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection', locs['collection_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
        


        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward (total):':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{'lstm actor loss:':>{pad}} {locs['lstm_actor_loss']:.4f}\n"""
                           f"""{'lstm encoder loss:':>{pad}} {locs['lstm_encoder_loss']:.4f}\n""")
        else:
            log_string = (f"""{'#' * width}\n""")
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        log_string += f"""{'-' * width}\n"""
        log_string += ep_string
        curr_it = locs['it'] - self.start_learning_iteration
        eta = self.tot_time / (curr_it + 1) * (locs['num_learning_iterations'] - curr_it)
        mins = eta // 60
        secs = eta % 60
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {mins:.0f} mins {secs:.1f} s\n""")
        print(log_string)




    def save(self, path, infos=None):
        state_dict = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            #'world_model_dict': self._world_model.state_dict(),
            #'wm_optimizer_state_dict': self._world_model._model_opt._opt.state_dict(),
            #'depth_predictor': self.depth_predictor.state_dict(),
            # 'discriminator_state_dict': self.alg.discriminator.state_dict(),
            # 'amp_normalizer': self.alg.amp_normalizer,
            'iter': self.current_learning_iteration,
            'infos': infos,
            } 

        if self.if_student:
            state_dict['lstm_encoder_state_dict'] = self.alg.lstm_encoder.state_dict()
            state_dict['lstm_actor_state_dict'] = self.alg.lstm_actor.state_dict()  

        torch.save(state_dict, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        # self.alg.discriminator.load_state_dict(loaded_dict['discriminator_state_dict'])        
        # self.alg.amp_normalizer = loaded_dict['amp_normalizer']

        if self.if_student:
            if 'lstm_encoder_state_dict' not in loaded_dict:
                warnings.warn("'lstm_encoder_state_dict' key does not exist, not loading lstm encoder...")
            else:
                print("Saved lstm encoder detected, loading...")
                self.alg.lstm_encoder.load_state_dict(loaded_dict['lstm_encoder_state_dict'])
            if 'lstm_actor_state_dict' in loaded_dict:
                print("Saved lstm actor detected, loading...")
                self.alg.lstm_actor.load_state_dict(loaded_dict['lstm_actor_state_dict'])
            else:
                print("No saved depth actor, Copying actor critic actor to depth actor...")
                self.alg.lstm_actor.load_state_dict(self.alg.actor_critic.state_dict())

        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_depth_actor_inference_policy(self, device=None):
        self.alg.lstm_actor.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.lstm_actor.to(device)
        return self.alg.lstm_actor.act_inference

    def get_depth_encoder_inference_policy(self, device=None):
        self.alg.lstm_encoder.eval()
        if device is not None:
            self.alg.lstm_encoder.to(device)
        return self.alg.lstm_encoder
