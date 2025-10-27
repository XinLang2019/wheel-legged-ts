
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[512, 256, 128],
                        critic_hidden_dims=[512, 256, 128],
                        activation='elu',
                        init_noise_std=1.0,
                        fixed_std=False,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        activation = get_activation(activation)

        self.num_actor_obs = num_actor_obs
        self.num_actions = num_actions

        #mlp_input_dim_a = num_one_step_obs + 3 + 16
        mlp_input_dim_c = num_critic_obs

        self.num_prop = 57
        self.num_scan = 187
        self.num_actions = num_actions
        self.privileged_dim = 29 # 6 + 23
        priv_encoder_dims = [64,32]
        scan_encoder_dims = [256,128]
        self.estimator_input_dim = self.privileged_dim
        self.history_dim = 42*5
        self.history_output = 32
        # Estimator
        #self.estimator = HIMEstimator(temporal_steps=self.history_size, num_one_step_obs=num_one_step_obs)

        # History Encoder
        # encoder_layers = []
        # encoder_layers.append(nn.Linear(self.history_dim, encoder_hidden_dims[0]))
        # encoder_layers.append(activation)
        # for l in range(len(encoder_hidden_dims)):
        #     if l == len(encoder_hidden_dims) - 1:
        #         encoder_layers.append(nn.Linear(encoder_hidden_dims[l], self.history_output))
        #     else:
        #         encoder_layers.append(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l + 1]))
        #         encoder_layers.append(activation)
        # self.history_encoder = nn.Sequential(*encoder_layers)




        self.estimator_output_dim = 8
        estimator_layers = []
        estimator_layers.append(nn.Linear(self.estimator_input_dim, priv_encoder_dims[0]))
        estimator_layers.append(activation)
        for l in range(len(priv_encoder_dims)):
            if l == len(priv_encoder_dims) - 1:
                estimator_layers.append(nn.Linear(priv_encoder_dims[l], self.estimator_output_dim))
            else:
                estimator_layers.append(nn.Linear(priv_encoder_dims[l], priv_encoder_dims[l + 1]))
                estimator_layers.append(activation)
        # estimator_layers.append(nn.Tanh())
        self.estimator = nn.Sequential(*estimator_layers)


        self.scan_output_dim = 16
        scan_layers = []
        scan_layers.append(nn.Linear(self.num_scan, scan_encoder_dims[0]))
        scan_layers.append(activation)
        for l in range(len(scan_encoder_dims)):
            if l == len(scan_encoder_dims) - 1:
                scan_layers.append(nn.Linear(scan_encoder_dims[l], self.scan_output_dim))
            else:
                scan_layers.append(nn.Linear(scan_encoder_dims[l], scan_encoder_dims[l + 1]))
                scan_layers.append(activation)
        # estimator_layers.append(nn.Tanh())
        self.scan_encoder = nn.Sequential(*scan_layers)

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(self.estimator_output_dim + self.scan_output_dim + self.num_prop, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
                #actor_layers.append(nn.Tanh())
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        #print(f'Estimator: {self.estimator.encoder}')

        # Action noise
        self.fixed_std = fixed_std
        std = init_noise_std * torch.ones(num_actions)
        self.std = torch.tensor(std) if fixed_std else nn.Parameter(std)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs,scandots_latent=None):
        if scandots_latent is None:
            scan_latent = self.scan_encoder(obs[:,self.num_prop+self.estimator_input_dim:self.num_prop + self.estimator_input_dim+ self.num_scan]) 
            estimator_latent = self.estimator(obs[:, :self.estimator_input_dim])
            all_latent = torch.cat([scan_latent,estimator_latent],dim=1)
        else:
            all_latent = scandots_latent
                
        prop =  obs[:,self.privileged_dim:self.privileged_dim+self.num_prop]

        backbone_input = torch.cat([prop,all_latent], dim=1)
        mean = self.actor(backbone_input)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
 

    def act_inference(self, obs,scandots_latent=None):
        #print(obs.shape)
        if scandots_latent is None:
            scan_latent = self.scan_encoder(obs[:,self.num_prop+self.estimator_input_dim:self.num_prop + self.estimator_input_dim+ self.num_scan]) #裁剪 最后的self.num_scan维 地形观测
            estimator_latent = self.estimator(obs[:, : self.estimator_input_dim]) # 裁剪 前self.estimator_input_dim维 特权观测
            all_latent = torch.cat([scan_latent,estimator_latent],dim=1)
        else:
            all_latent = scandots_latent
                               
        prop =  obs[:,self.privileged_dim:self.privileged_dim+self.num_prop] # 裁剪本体感知观测

        backbone_input = torch.cat([prop, all_latent], dim=1)
        mean = self.actor(backbone_input)
        return mean
    
    def infer_latent(self,obs):
        scan_latent = self.scan_encoder(obs[:,self.num_prop+self.estimator_input_dim:self.num_prop + self.estimator_input_dim+ self.num_scan]) 
        estimator_latent = self.estimator(obs[:, : self.estimator_input_dim])
        all_latent = torch.cat([scan_latent,estimator_latent],dim=1)
        return all_latent    

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
    


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
