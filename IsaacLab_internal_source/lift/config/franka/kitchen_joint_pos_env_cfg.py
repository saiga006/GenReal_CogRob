# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Kitchen environment configuration for the Franka robot with joint position control.
"""

from copy import deepcopy
import os
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import TiledCameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg, TerminationTermCfg as DoneTerm, EventTermCfg as EventTerm
from isaaclab.utils.math import transform_points
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils
import torch
import os

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG, FRANKA_PANDA_HIGH_PD_CFG  # isort: skip
from .kitchen_scene import MinimalKitchenSceneCfg  # isort: skip

KITCHEN_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "kinova", "assets")


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
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # state observations
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel)
        object = ObservationTermCfg(func=mdp.object_position_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("object")})
        actions = ObservationTermCfg(func=mdp.last_action)
        wrist_cam = ObservationTermCfg(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist_camera"), "data_type": "rgb", "normalize": False}
        )
       # wrist_cam_depth = ObservationTermCfg(
       #     func=mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist_camera"), "data_type": "distance_to_image_plane", "normalize": False}
       # )
        eef_pos = ObservationTermCfg(func=mdp.ee_frame_pos)
        eef_quat = ObservationTermCfg(func=mdp.ee_frame_quat)
        gripper_pos = ObservationTermCfg(func=mdp.gripper_pos)

        def __post_init__(self):
            """Post-initialization."""
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class FrankaKitchenLiftEnvCfg(LiftEnvCfg):
    # The active observations for the environment
    observations: ObservationsCfg = ObservationsCfg()
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Add light intensity randomization on reset
        self.events.randomize_light = EventTerm(
            func=mdp.randomize_light_intensity, 
            mode="reset",
            params={
                "light_path": "/World/light",
                "intensity_range": (700.0, 2000.0)  # Randomize between 700-1300 (centered around 1000)
            }
        )

        # List of image observations in policy observations
        self.image_obs_list = ["wrist_cam", "wrist_cam_depth"]

        # Set settings for camera rendering
        self.rerender_on_reset = True
        self.sim.render.antialiasing_mode = "OFF"  # disable dlss

        # Set kitchen scene with single environment
        self.scene = MinimalKitchenSceneCfg(num_envs=1, env_spacing=2.5)

        # Set Franka as robot
        robot_cfg = deepcopy(FRANKA_PANDA_HIGH_PD_CFG)
        robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
        robot_cfg.init_state.pos = (1.9, 1.25, 0.8)
        robot_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        # override the default initial state
        robot_cfg.init_state.joint_pos = {
            "panda_joint1": 0.0,
            "panda_joint2": -0.8,
            "panda_joint3": 0.0,
            "panda_joint4": -2.3,
            "panda_joint5": 0.0,
            "panda_joint6": 1.5,
            "panda_joint7": 0.8,
            "panda_finger_joint1": 0.04,
            "panda_finger_joint2": 0.04,
        }
        self.scene.robot = robot_cfg

        # Set the object (using the CONTAINER_CFG from kitchen scene as the liftable object)
        self.scene.object = self.scene.CONTAINER_CFG

        # Create a camera config for the wrist camera
        wrist_camera_cfg = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link7/camera",
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2.0)
            ),
            width=88, # Set desired policy input width
            height=88, # Set desired policy input height
            offset=CameraCfg.OffsetCfg(pos=(0.05639, -0.05639, -0.00305), rot=(0.0, 0.0, 0.0, -1.0), convention="ros"),
            data_types=["rgb", "distance_to_image_plane"],  # Added depth information
            update_period=0.0,  # Ensures camera updates every frame
        )
        self.scene.wrist_camera = wrist_camera_cfg

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "panda_hand"

        # Listens to the required transforms
        marker_cfg = deepcopy(FRAME_MARKER_CFG)
        # change marker scale
        marker_cfg.markers["frame"] = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd", scale=(0.1, 0.1, 0.1)
        )
        marker_cfg.prim_path = "/Visuals/FrameTransformer"

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )
        
        # Override the command configuration for teleoperation
        self.commands.object_pose.resampling_time_range = (999999, 999999)  # Almost never change
        self.commands.object_pose.debug_vis = False  # Hide visual markers
        # Fix target to a single position if desired
        self.commands.object_pose.ranges = mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.5, 0.5), pos_y=(0.0, 0.0), pos_z=(0.7, 0.7),
            roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        )

        # Set initial state of the microwave door to be open
        if hasattr(self.scene, "microwave"):
            self.scene.microwave.init_state.joint_pos = {"microjoint": -1.57}
        
        # Set initial state of the fridge door to be open
        if hasattr(self.scene, "fridge"):
            self.scene.fridge.init_state.joint_pos = {"fridge_door_joint": 1.57}

        # Define success condition for recording demonstrations
        self.terminations.success = DoneTerm(
            func=object_in_microwave_and_hand_out,
            params={
                "object_name": "object",
                "microwave_name": "microwave",
                "hand_body_name": "panda_hand",
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
class FrankaKitchenLiftEnvCfg_PLAY(FrankaKitchenLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
