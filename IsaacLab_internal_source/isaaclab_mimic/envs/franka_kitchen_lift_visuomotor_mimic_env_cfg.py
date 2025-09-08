# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Isaac Lab Mimic environment config class for Franka Kitchen Lift Visuomotor task.
"""

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift.config.franka.kitchen_ik_rel_env_cfg import (
    FrankaKitchenLiftEnvCfg,
)


@configclass
class FrankaKitchenLiftVisuomotorMimicEnvCfg(FrankaKitchenLiftEnvCfg, MimicEnvCfg):
    """
    Isaac Lab Mimic environment config class for Franka Kitchen Lift Visuomotor env.
    """

    def __post_init__(self):
        # post init of parent
        #FrankaKitchenLiftEnvCfg.__post_init__(self)
        #MimicEnvCfg.__post_init__(self)
        super().__post_init__()
        # Add the keys for low-dimensional observations to be saved in the dataset
        self.datagen_config.low_dim_obs_keys = ["eef_pos", "eef_quat", "gripper_pos", "joint_pos", "object", "joint_vel", "actions"]

        # Override the existing values for data generation configuration
        self.datagen_config.name = "isaac_lab_franka_kitchen_lift_visuomotor_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        # The following are the subtask configurations for the kitchen task.
        subtask_configs = []
        
        # Subtask 1: Move to fridge
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to the tomato can object
                object_ref="object",
                # This key corresponds to the binary indicator in "datagen_info" that signals
                # when this subtask is finished (e.g., on a 0 to 1 edge).
                subtask_term_signal="move_to_fridge",
                # Specifies time offsets for data generation when splitting a trajectory into
                # subtask segments. Random offsets are added to the termination boundary.
                subtask_term_offset_range=(5, 15),
                # Selection strategy for the source subtask segment during data generation
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.02,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=5,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
                # Description of the subtask
                description="Move robot end-effector to the fridge area",
                # Instructions for the next subtask
                next_subtask_description="Grasp the tomato can from inside the fridge",
            )
        )
        
        # Subtask 2: Grasp tomato can
        subtask_configs.append(
            SubTaskConfig(
                # Object involved in this subtask
                object_ref="object",
                # Corresponding key for the binary indicator in "datagen_info" for completion
                subtask_term_signal="grasp_tomato_can",
                # Time offsets for data generation when splitting a trajectory
                subtask_term_offset_range=(5, 15),
                # Selection strategy for source subtask segment
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.02,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=5,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
                # Description of the subtask
                description="Grasp the tomato can with the gripper",
                # Instructions for the next subtask
                next_subtask_description="Move the grasped tomato can to the microwave",
            )
        )
        
        # Subtask 3: Move to microwave
        subtask_configs.append(
            SubTaskConfig(
                # Object involved in this subtask
                object_ref="object",
                # Corresponding key for the binary indicator in "datagen_info" for completion
                subtask_term_signal="move_to_microwave",
                # Time offsets for data generation when splitting a trajectory
                subtask_term_offset_range=(5, 15),
                # Selection strategy for source subtask segment
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.02,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=5,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
                # Description of the subtask
                description="Move the grasped tomato can to the microwave entrance",
                # Instructions for the next subtask
                next_subtask_description="Place the tomato can inside the microwave cavity",
            )
        )
        
        # Subtask 4: Place in microwave
        subtask_configs.append(
            SubTaskConfig(
                # Object involved in this subtask
                object_ref="object",
                # Corresponding key for the binary indicator in "datagen_info" for completion
                subtask_term_signal="place_in_microwave",
                # Time offsets for data generation when splitting a trajectory
                subtask_term_offset_range=(5, 15),
                # Selection strategy for source subtask segment
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.02,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=5,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
                # Description of the subtask
                description="Place the tomato can inside the microwave cavity",
                # Instructions for the next subtask
                next_subtask_description="Move the robot hand away from the microwave",
            )
        )
        
        # Subtask 5: Move away from microwave (final subtask)
        subtask_configs.append(
            SubTaskConfig(
                # Object involved in this subtask
                object_ref="object",
                # End of final subtask does not need to be detected
                subtask_term_signal=None,
                # No time offsets for the final subtask
                subtask_term_offset_range=(0, 0),
                # Selection strategy for source subtask segment
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.02,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=5,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
                # Description of the subtask
                description="Move the robot hand away from the microwave cavity",
                # Instructions for the next subtask
                next_subtask_description="Task complete",
            )
        )
        
        # Assign subtask configurations to the end effector
        self.subtask_configs["end_effector"] = subtask_configs
