# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import os
from copy import deepcopy

import isaaclab.sim as sim_utils
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp
from .cognitive_robotics_genreal_env_cfg import CognitiveRoboticsGenrealEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.kinova import KINOVA_GEN3_N7_ROBOTIQ_2F85_HIGH_PD_CFG  # isort: skip


@configclass
class KinovaTeleopEnvCfg(CognitiveRoboticsGenrealEnvCfg):
    """Configuration for teleoperation environment using Kinova robot and kitchen scene."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Get the base path for assets
        KITCHEN_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

        # Set Kinova as robot
        robot_cfg = deepcopy(KINOVA_GEN3_N7_ROBOTIQ_2F85_HIGH_PD_CFG)
        robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
        # spawn the robot using the custom USD
        robot_cfg.spawn = sim_utils.UsdFileCfg(
            usd_path=f"{KITCHEN_ASSETS_DIR}/Kinova/kinova_gen3_robotiq_2f_85_working.usd",
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=16, solver_velocity_iteration_count=1
            ),
        )
        # override the default initial state
        robot_cfg.init_state.joint_pos = {
            "joint_1": 0.0, "joint_2": 0.65, "joint_3": 0.0, "joint_4": 1.89,
            "joint_5": 0.0, "joint_6": 0.6, "joint_7": -1.57,
        }
        self.scene.robot = robot_cfg

        # Listens to the required transforms
        marker_cfg = deepcopy(FRAME_MARKER_CFG)
        # change marker scale
        marker_cfg.markers["frame"] = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd", scale=(0.1, 0.1, 0.1)
        )
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/end_effector_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.1034),
                        rot=(math.pi, 0.0, 0.0),
                    ),
                ),
            ],
        )

        self.commands.object_pose.body_name = "end_effector_link"

        # Set IK control for teleoperation
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["joint_.*"],
            body_name="end_effector_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )





@configclass
class KinovaTeleopEnvCfg_PLAY(KinovaTeleopEnvCfg):
    """Configuration for teleoperation play mode."""
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # Play mode settings
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 600.0  # 10 minutes for extended play
        
        # Disable randomization for play mode
        self.observations.policy.enable_corruption = False
