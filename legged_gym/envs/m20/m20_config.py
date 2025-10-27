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
import glob

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class M20RoughWCfg( LeggedRobotCfg ):

    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_actions = 16
        include_history_steps = None  # Number of steps of history to include.
        num_prop = 57
        num_privileged = 29
        num_scan = 187
        num_observations = num_prop + num_privileged + num_scan
        num_privileged_obs = num_observations
        reference_state_initialization = False
        reference_state_initialization_prob = 0.85

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'fl_hipx_joint': 0.0,   # [rad]
            'hl_hipx_joint': 0.0,   # [rad]
            'fr_hipx_joint': -0.0 ,  # [rad]
            'hr_hipx_joint': -0.0,   # [rad]

            'fl_hipy_joint': -0.6,   # [rad]
            'hl_hipy_joint':  0.6,   # [rad]
            'fr_hipy_joint': -0.6 ,  # [rad]
            'hr_hipy_joint':  0.6,   # [rad]

            'fl_knee_joint':  1.0,   # [rad]
            'hl_knee_joint': -1.0,   # [rad]
            'fr_knee_joint':  1.0 ,  # [rad]
            'hr_knee_joint': -1.0,   # [rad]

            'fl_wheel_joint':  0.0,   # [rad]
            'hl_wheel_joint':  0.0,   # [rad]
            'fr_wheel_joint': -0.0 ,  # [rad]
            'hr_wheel_joint': -0.0,   # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hipx_joint': 70.,'hipy_joint': 70.,'knee_joint': 70.,"wheel_joint":0}  # [N*m/rad]
        damping = {'hipx_joint': 5,'hipy_joint': 5,'knee_joint': 5,"wheel_joint":2}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        vel_scale = 5 # 5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        wheel_speed = 1

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'trimesh'   # plane、trimesh
        measure_heights = True

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/m20/urdf/m20.urdf'
        name = "m20"
        foot_name = "wheel"
        wheel_name =["wheel"]
        penalize_contacts_on = ["hipy", "knee"]
        # terminate_after_contacts_on = [
        #     "base", "FL_calf", "FR_calf", "RL_calf", "RR_calf",
        #     "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        terminate_after_contacts_on = ["base_link"]
        replace_cylinder_with_capsule = False
        flip_visual_attachments = False

    class domain_rand:
        randomize_friction = True
        friction_range = [0.25, 2.0]
        randomize_restitution = True
        restitution_range = [0.0, 0.0]        
        randomize_base_mass = True
        added_mass_range = [-1., 2.]
        randomize_link_mass = True
        link_mass_range = [0.8, 1.2]
        randomize_com_pos = True
        com_x_pos_range = [-0.05, 0.05]
        com_y_pos_range = [-0.05, 0.05]
        com_z_pos_range = [-0.05, 0.05]

        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0

        randomize_gains = True
        stiffness_multiplier_range = [0.8, 1.2]
        damping_multiplier_range = [0.8, 1.2]

        randomize_torque = True
        torque_multiplier_range = [0.8, 1.2]

        # 信号延迟
        add_lag = False
        randomize_lag_timesteps = True # 随机延迟步数
        randomize_lag_timesteps_perstep = False
        lag_timesteps_range = [0, 4]

        # 关节库伦、粘性摩擦
        randomize_coulomb_friction = False
        joint_coulomb_range = [0.1, 0.9]
        joint_viscous_range = [0.05, 0.1]

        # 关节摩擦
        randomize_joint_friction = False
        randomize_joint_friction_each_joint = False # 每个单独配置
        joint_friction_range = [0.01, 0.5]   #multiplier
    
        # 关节阻尼
        randomize_joint_damping = False
        randomize_joint_damping_each_joint = False  # 每个单独配置
        joint_damping_range = [0.6, 1.2]       #multiplier

        # 关节电枢
        randomize_joint_armature = False
        randomize_joint_armature_each_joint = False  # 每个单独配置
        joint_armature_range = [0.0001, 0.05]     # Factor
        joint_1_armature_range = [0.14, 0.16]
        joint_2_armature_range = [0.14, 0.16]
        joint_3_armature_range = [0.34, 0.36]
        joint_4_armature_range = [0.14, 0.16]
        joint_5_armature_range = [0.14, 0.16]
        joint_6_armature_range = [0.34, 0.36]
        joint_7_armature_range = [0.14, 0.16]
        joint_8_armature_range = [0.14, 0.16]
        joint_9_armature_range = [0.34, 0.36]
        joint_10_armature_range = [0.14, 0.16]   
        joint_11_armature_range = [0.14, 0.16]
        joint_12_armature_range = [0.34, 0.36] 

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.0
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.0

    class rewards( LeggedRobotCfg.rewards ):
        only_positive_rewards = True
        soft_dof_pos_limit = 0.9
        base_height_target = 0.52
        tracking_sigma = 0.4
        # wheel vel Penalize
        velocity_threshold = 0.15 # 实际机身速度多大时算“已经在动” 0.5
        command_threshold = 0.1 # 控制命令多大时算“开始移动” 0.1
        stand_still_scale = 5

        joint_limit_lower = [-0.2,0.4,-2.5,0.0,
                             -0.3,0.4,-2.5,0.0,
                             -0.2,0.4,-2.5,0.0,
                             -0.3,0.4,-2.5,0.0]
        
        joint_limit_upper = [0.3,2.5,-1.0,0.0,
                             0.2,2.5,-1.0,0.0,
                             0.3,2.5,-1.0,0.0,
                             0.2,2.5,-1.0,0.0]
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -0.0
            tracking_lin_vel = 4.0
            tracking_ang_vel = 2.0
            lin_vel_z = -1.0
            ang_vel_xy = -0.05 # 可能影响爬高台
            orientation = -0.2 # 复杂地形不能给太大，会影响机身适应地形倾角
            torques = -0.0001
            dof_vel = 0
            dof_acc = -1e-7
            dof_acc_wheel = -3e-7 # 惩罚轮子加速度
            base_height = -1.0 
            feet_air_time =  -0.0
            collision = -0.1
            feet_stumble = -0.0
            action_rate = -0.0002
            stand_still = -0.01
            dof_pos_limits = -0.0
            hip_action_l2 = -0.0
            hip_default = -0.5
            dof_error = -0.15
            soft_joint_pos_limit = -0.0
            feet_contact = 0.0

    class commands:
        curriculum = True
        max_curriculum = 1.0
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error 需要开启，有利于训练地形等级
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-1.57, 1.57]    # min max [rad/s]
            heading = [-0, 0] # 让机器人直走

class M20RoughCfgPPO( LeggedRobotCfgPPO ):
    runner_class_name = 'OnPolicyRunner'
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'm20'
        algorithm_class_name = 'PPO'
        policy_class_name = 'ActorCritic'
        max_iterations = 30000 # number of policy updates
        num_steps_per_env = 24 # per iteration
        save_interval = 500

        min_normalized_std = [0.05, 0.02, 0.05, 0.001] * 4

        if_student = False
        
    class LSTMEncoder:
        input_size = 57
        hidden_size = 256
        num_layers = 3
        learning_rate = 1e-3
        num_steps_per_env = 50           
  