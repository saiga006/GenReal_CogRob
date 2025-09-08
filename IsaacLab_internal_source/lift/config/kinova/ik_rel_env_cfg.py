# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.kinova import KINOVA_GEN3_N7_HIGH_PD_CFG  # isort: skip


@configclass
class KinovaCubeLiftEnvCfg(joint_pos_env_cfg.KinovaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Kinova as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.

        # -- 1. Disable random command generation for teleoperation
        # Set a very long resampling time to prevent the target from changing.
        self.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        # Hide the distracting command visualization.
        self.commands.object_pose.debug_vis = False
        self.commands.object_pose.body_name = "end_effector_link"

        # -- 2. Set actions for intuitive teleoperation
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["joint_[1-7]"],
            body_name="end_effector_link",
            # Use absolute mode for intuitive control relative to the robot's base.
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=2.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.1034]),
        )


@configclass
class KinovaCubeLiftEnvCfg_PLAY(KinovaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
