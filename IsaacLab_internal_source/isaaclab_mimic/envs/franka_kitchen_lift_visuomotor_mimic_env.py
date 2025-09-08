# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Isaac Lab Mimic environment wrapper class for Franka Kitchen Lift Visuomotor task.
This version is adapted for an IK-relative control environment.
"""

import torch
from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv


class FrankaKitchenLiftVisuomotorMimicEnv(ManagerBasedRLMimicEnv):
    """
    Isaac Lab Mimic environment wrapper class for Franka Kitchen Lift IK Rel Visuomotor env.
    """

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """
        Get current robot end effector pose. Should be the same frame as used by the robot end-effector controller.
        """
        if env_ids is None:
            env_ids = slice(None)

        # Retrieve end effector pose from the observation buffer, as defined in the env config.
        eef_pos = self.obs_buf["policy"]["eef_pos"][env_ids]
        eef_quat = self.obs_buf["policy"]["eef_quat"][env_ids]
        # Quaternion format from obs_buf is w,x,y,z which is what make_pose expects.
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    def get_object_poses(self, object_name: str = "object", env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Required by the mimic framework."""
        if env_ids is None:
            env_ids = slice(None)

        # Get object position and orientation in world frame
        obj_pos_w = self.scene[object_name].data.root_pos_w[env_ids]
        obj_quat = self.scene[object_name].data.root_quat_w[env_ids]
        
        # Convert to environment frame by subtracting environment origins
        obj_pos = obj_pos_w - self.scene.env_origins[env_ids]

        # Convert to homogeneous transformation matrix
        obj_pose_matrix = PoseUtils.make_pose(obj_pos, PoseUtils.matrix_from_quat(obj_quat))
        return {object_name: obj_pose_matrix}

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """
        Takes a target pose and gripper action for the end effector controller and returns a normalized
        delta pose action to try and achieve that target pose.
        """
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        # target position and rotation
        (target_eef_pose,) = target_eef_pose_dict.values()
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

        # current position and rotation
        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        # normalized delta position action
        delta_position = target_pos - curr_pos

        # normalized delta rotation action
        delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
        delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
        delta_rotation = PoseUtils.axis_angle_from_quat(delta_quat)

        # get gripper action for single eef
        (gripper_action,) = gripper_action_dict.values()

        # add noise to action
        pose_action = torch.cat([delta_position, delta_rotation], dim=0)
        if action_noise_dict is not None:
            noise = action_noise_dict[eef_name] * torch.randn_like(pose_action)
            pose_action += noise
            pose_action = torch.clamp(pose_action, -1.0, 1.0)

        # The total action is the 6D delta pose plus the 1D gripper action
        return torch.cat([pose_action, gripper_action], dim=0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Converts a delta-pose action to a target pose for the end effector controller.
        Inverse of @target_eef_pose_to_action.
        """
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        # The action is composed of delta position and delta rotation
        delta_position = action[:, :3]
        delta_rotation = action[:, 3:6]

        # current position and rotation
        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=None)
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        # get pose target by applying the delta
        target_pos = curr_pos + delta_position

        # Convert delta_rotation from axis-angle to rotation matrix with zero rotation handling
        delta_rotation_angle = torch.linalg.norm(delta_rotation, dim=-1, keepdim=True)
        delta_rotation_axis = delta_rotation / delta_rotation_angle
        
        # Handle near-zero rotation angles to avoid numerical instability
        is_close_to_zero_angle = torch.isclose(delta_rotation_angle, torch.zeros_like(delta_rotation_angle)).squeeze(-1)
        delta_rotation_axis[is_close_to_zero_angle] = torch.zeros_like(delta_rotation_axis)[is_close_to_zero_angle]
        
        delta_quat = PoseUtils.quat_from_angle_axis(delta_rotation_angle.squeeze(-1), delta_rotation_axis)
        delta_rot_mat = PoseUtils.matrix_from_quat(delta_quat)
        target_rot = torch.matmul(delta_rot_mat, curr_rot)

        target_poses = PoseUtils.make_pose(target_pos, target_rot).clone()

        return {eef_name: target_poses}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Extracts the gripper actuation part from a sequence of env actions.
        The arm action is 6D, so the gripper action is the last dimension.
        """
        eef_name = list(self.cfg.subtask_configs.keys())[0]
        # last dimension is gripper action
        return {eef_name: actions[..., -1:]}
    """
    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        
        Returns a dictionary of subtask termination signals for each subtask.
        These signals are used by the datagen framework to determine when a subtask is complete.
        
        if env_ids is None:
            env_ids = slice(None)

        subtask_term_signals = {}
        for subtask_cfg in self.cfg.subtask_configs.values():
            for subtask in subtask_cfg:
                if subtask.subtask_term_signal is not None:
                    subtask_term_signals[subtask.subtask_term_signal] = self.event_manager.is_terminated(
                        subtask.subtask_term_signal
                    )[env_ids]
        return subtask_term_signals
    """