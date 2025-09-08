# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Lift environment configuration for the Kinova Gen3 robot with joint position control.
"""

from copy import deepcopy
import os
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

import torch
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.math import transform_points

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.kinova import KINOVA_GEN3_N7_ROBOTIQ_2F85_HIGH_PD_CFG  # isort: skip
from .kitchen_scene import MinimalKitchenSceneCfg  # isort: skip

KITCHEN_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")


def object_in_microwave_and_hand_out(
    env, object_name: str, microwave_name: str, hand_body_name: str, microwave_box_dims: tuple[float, float, float]
):
    """
    Checks if the object is inside the microwave's bounding box and the robot's hand is outside.
    """
    # Get assets from the scene
    obj = env.scene[object_name]
    microwave = env.scene[microwave_name]
    robot = env.scene["robot"]

    # Get the world poses
    microwave_pos_w, microwave_quat_w = microwave.data.root_pos_w, microwave.data.root_quat_w
    obj_pos_w = obj.data.root_pos_w
    
    # Get robot hand position
    hand_link_idx = robot.body_names.index(hand_body_name)
    hand_pos_w = robot.data.body_pos_w[:, hand_link_idx]
    
    # Transform object position to microwave's local frame
    obj_pos_local = transform_points(obj_pos_w.unsqueeze(1), microwave_pos_w, microwave_quat_w).squeeze(1)
    hand_pos_local = transform_points(hand_pos_w.unsqueeze(1), microwave_pos_w, microwave_quat_w).squeeze(1)
    
    # Define the microwave's interior bounding box (centered at origin in local frame)
    box_min = torch.tensor([-microwave_box_dims[0] / 2, -microwave_box_dims[1] / 2, 0.0], device=env.device)
    box_max = torch.tensor([microwave_box_dims[0] / 2, microwave_box_dims[1] / 2, microwave_box_dims[2]], device=env.device)
    
    # Check if object is inside the box
    is_obj_in = torch.all(
        (obj_pos_local >= box_min) & (obj_pos_local <= box_max), dim=1
    )
    
    # Check if hand is outside the box
    is_hand_in = torch.all(
        (hand_pos_local >= box_min) & (hand_pos_local <= box_max), dim=1
    )
    is_hand_out = ~is_hand_in

    # Combine the conditions: success = object is in AND hand is out
    success = is_obj_in & is_hand_out
    return success.unsqueeze(1)


@configclass
class KinovaCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set kitchen scene with single environment
        self.scene = MinimalKitchenSceneCfg(num_envs=1,env_spacing=2.5)

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
        robot_cfg.init_state.pos=(2.15, 1.25, 0.8)
        robot_cfg.init_state.rot=(1.0, 0.0, 0.0, 0.0)  # 90-degree rotation around Z-axis
        # override the default initial state
        robot_cfg.init_state.joint_pos = {
            "joint_1": 0.0,
            "joint_2": 0.0,
            "joint_3": 0.0,
            "joint_4": 0.0,
            "joint_5": 0.0,
            "joint_6": 0.0,
            "joint_7": 0.0,
        }
        self.scene.robot = robot_cfg

        # Set the object (using the CONTAINER_CFG from kitchen scene as the liftable object)
        self.scene.object = self.scene.CONTAINER_CFG

        # Set actions for the specific robot type (kinova)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["joint_[1-7]"], scale=0.8, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger_joint"],
            open_command_expr={"finger_joint": 0.8},  # Increased opening for Robotiq 2F-85
            close_command_expr={"finger_joint": 0.0},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "end_effector_link"

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
                    prim_path="{ENV_REGEX_NS}/Robot/gripper_base_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.1034),
                    ),
                ),
            ],
        )
        # Override the command configuration for teleoperation
        self.commands.object_pose.resampling_time_range = (999999, 999999)  # Almost never change
        self.commands.object_pose.debug_vis = False  # Hide visual markers
        # Fix target to a single position if desired
        self.commands.object_pose.ranges = mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.5, 0.5), pos_y=(0.0, 0.0), pos_z=(0.4, 0.4),
            roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        )

        # Set initial state of the microwave door to be open
        # Based on the user's request, 'micro_joint' is assumed to be the door joint.
        # Setting to -1.57 radians (approx. -90 degrees) to open it to its full extent.
        if hasattr(self.scene, "microwave"):
            self.scene.microwave.init_state.joint_pos = {"microjoint": -1.57}
        # Set initial state of the fridge door to be open
        if hasattr(self.scene, "fridge"):
            self.scene.fridge.init_state.joint_pos = {"fridge_door_joint": 1.57}

        # Add a camera to the wrist
        self.scene.wrist_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/bracelet_link/wrist_camera",
            spawn=sim_utils.PinholeCameraCfg(),
            update_period=0.1,
            height=240,
            width=320,
            data_types=["rgb"], # You can also add "depth"
            offset=CameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.0), rot=(0.0316, -0.220, -0.974, 0.0415), convention="ros")
        )

        # Define success condition for recording demonstrations
        self.terminations.success = DoneTerm(
            func=object_in_microwave_and_hand_out,
            params={
                "object_name": "object",
                "microwave_name": "microwave", 
                "hand_body_name": "end_effector_link",
                # These dimensions define the "inside" of the microwave.
                # You may need to tune these values for your specific microwave asset.
                "microwave_box_dims": (0.3, 0.3, 0.25),
            },
        )

        # Add a visual marker for the microwave bounding box
        # This adds a translucent green box that represents the success area.
        self.scene.microwave_bbox_marker = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/SuccessBBox",
            spawn=sim_utils.CuboidCfg(
                size=(0.3, 0.3, 0.25),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0), 
                    opacity=0.3,
                    metallic=0.0,
                    roughness=1.0
                ),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(2.3, 0.5, 1.25), # Position it inside the microwave cavity
            )
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
