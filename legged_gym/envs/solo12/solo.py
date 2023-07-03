
from time import time, sleep
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot
from legged_gym.utils.math import wrap_to_pi
from .solo12_config import Solo12Cfg, Solo12CfgPPO

class Solo12(LeggedRobot):

    cfg: Solo12Cfg
    cfg_ppo = Solo12CfgPPO()

    def _init_buffers(self):
        super()._init_buffers()
        
        # q_target(t-2)
        self.last_last_q_target = torch.zeros(self.num_envs, self.num_dof, device=self.device, requires_grad=False)
        # q_target(t-1)
        self.last_q_target = torch.zeros(self.num_envs, self.num_dof, device=self.device, requires_grad=False)
        # q_target(t)
        self.q_target = torch.zeros(self.num_envs, self.num_dof, device=self.device, requires_grad=False)
        
        self.last_last_q_target[:] = self.default_dof_pos
        self.last_q_target[:] = self.default_dof_pos
        self.q_target[:] = self.default_dof_pos

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.last_last_q_target[env_ids] = self.default_dof_pos
        self.last_q_target[env_ids] = self.default_dof_pos
        self.q_target[env_ids] = self.default_dof_pos

    def check_termination(self):
        super().check_termination()
        self.reset_buf |= torch.abs(self.roll[:]) > 2
        # HACK: this partially fixes contact on base/truck not being detected (why?)        

    def _post_physics_step_callback(self):
        self.last_last_q_target[:] = self.last_q_target
        self.last_q_target[:] = self.q_target
        self.q_target[:] = self._get_q_target(self.actions)

        self.roll, self.pitch = self._get_roll_pitch()

        super()._post_physics_step_callback()
       
    def _get_q_target(self, actions):
        return self.cfg.control.action_scale * actions + self.default_dof_pos
     
    def _get_roll_pitch(self):
        roll, pitch, _ = get_euler_xyz(self.root_states[:, 3:7])
        roll, pitch = wrap_to_pi(roll), wrap_to_pi(pitch)
        return roll, pitch

    # --- rewards (see paper) ---

    def _reward_velocity(self):
        """
        Rewards robot for going at the speed set by the virtual joystick of the user. 
        """
        v_speed = torch.hstack((self.base_lin_vel[:, :2], self.base_ang_vel[:, 2:3]))
        vel_error = torch.sum(torch.square(self.commands[:, :3] - v_speed), dim=1)
        return torch.exp(-vel_error)
    
    def _reward_foot_clearance(self):
        """
        Unsure how it is implemented. What it is trying to do is reward robot for lifting its foot high off the ground 
        when walking. 
        """
        feet_z = self.get_feet_height()  # Get height of feet from ground
        height_err = torch.square(feet_z - self.cfg.control.feet_height_target)
        feet_speed = torch.sum(torch.square(self.body_state[:, self.feet_indices, 7:9]), dim=2)
        return torch.sum(height_err * torch.sqrt(feet_speed), dim=1)

    def _reward_foot_slip(self):
        # inspired from LeggedRobot::_reward_feet_air_time
        speed = torch.sum(torch.square(self.body_state[:, self.feet_indices, 7:9]), dim=2)

        return torch.sum(self.filtered_feet_contacts * speed, dim=1)

    def _reward_test(self):
        """P
        In order to define a new reward function called "test" we first have to define the reward scale in the "scales()" function 
        in the "solo12_config.py" file.
        """
        r = self.base_lin_vel[:, 2]
        return r
    
    def _reward_vel_z(self):
        """P
        Rewards the robot if its base has a high linear velocity. 

        As reward has to be scalar, "base_lin_vel" must be tensor where the first dimension only has one component. 

        We could encourage jumping by rewarding the robot if its base has a high linear velocity in the upwards direction.
        To see what direction up is, we can have a look at the gravity vector. 

        base_lin_vel[0] is linear velocity in the forwards/backwards direction. If it is equal to -2, this means robot is moving backwards.
        base_lin_vel[1] is linear velocity in the lateral direction.

        """
        r = torch.square(self.base_lin_vel[:, 2])
        return r
   
    def _reward_roll_pitch(self):
        return torch.sum(torch.square(torch.stack((self.roll, self.pitch), dim=1)), dim=1)
    
    def _reward_joint_pose(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
    
    def _reward_power_loss(self):
        TAU_U = 0.0477 # [Nm]
        B = 0.000135 # [Nm.s]
        K = 4.81 # [Nms]

        q_dot = self.dof_vel
        tau_f = TAU_U * torch.sign(q_dot) + B * q_dot
        P_f = tau_f * q_dot
        P_j = torch.square(self.torques + tau_f) / K

        return torch.sum(P_f + P_j, dim=1)
    
    def _reward_smoothness_1(self):
        return torch.sum(torch.square(self.q_target - self.last_q_target), dim=1)
    
    def _reward_smoothness_2(self):
        return torch.sum(torch.square(self.q_target - 2 * self.last_q_target + self.last_last_q_target), dim=1)