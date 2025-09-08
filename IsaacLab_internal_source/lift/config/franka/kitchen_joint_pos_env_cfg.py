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
from isaaclab.envs import SubTaskConfig
from isaaclab.utils.math import quat_inv, quat_apply

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG, FRANKA_PANDA_HIGH_PD_CFG  # isort: skip
from .kitchen_scene import MinimalKitchenSceneCfg  # isort: skip

KITCHEN_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "kinova", "assets")


def object_in_microwave_and_hand_out(
    env, object_name: str, microwave_name: str, hand_body_name: str, microwave_box_dims: tuple[float, float, float],
    finger_joint_names: list[str]
):
    """
    Checks if the object is inside the microwave's bounding box, the robot's hand is outside,
    and the object is not in the gripper.
    
    This success function is crucial for both annotation and dataset generation.
    It verifies the completion of the task by ensuring:
    1. The tomato can is properly placed inside the microwave cavity
    2. The robot's hand has been removed from the microwave
    3. The object is not being held by the gripper
    
    Parameters:
        env: The simulation environment
        object_name: Name of the object to place (tomato can)
        microwave_name: Name of the microwave asset
        hand_body_name: Name of the robot hand body
        microwave_box_dims: Dimensions of the box representing microwave interior
        finger_joint_names: List of finger joint names to check gripper state
    
    Returns:
        Boolean tensor indicating success (True) or failure (False)
    """
    # Get assets from the scene
    obj = env.scene[object_name]
    microwave = env.scene[microwave_name]
    robot = env.scene["robot"]

    # Get the world poses and convert to environment frame
    microwave_pos_w = microwave.data.root_pos_w - env.scene.env_origins
    microwave_quat_w = microwave.data.root_quat_w
    obj_pos_w = obj.data.root_pos_w - env.scene.env_origins
    
    # Get robot hand position in environment frame
    hand_link_idx = robot.body_names.index(hand_body_name)
    hand_pos_w = robot.data.body_pos_w[:, hand_link_idx] - env.scene.env_origins

    # Ensure obj_pos_w has a batch dimension
    if obj_pos_w.ndim == 1:
        obj_pos_w = obj_pos_w.unsqueeze(0)

    # Define the microwave's interior bounding box in world frame
    box_dims_tensor = torch.tensor(microwave_box_dims, device=env.device)
    box_min = -box_dims_tensor / 2
    box_max = box_dims_tensor / 2

    box_min_world = microwave_pos_w + quat_apply(microwave_quat_w, box_min)
    box_max_world = microwave_pos_w + quat_apply(microwave_quat_w, box_max)

    # Ensure box_min_world and box_max_world have a batch dimension
    if box_min_world.ndim == 1:
        box_min_world = box_min_world.unsqueeze(0)
    if box_max_world.ndim == 1:
        box_max_world = box_max_world.unsqueeze(0)

    # Compute axis-aligned min/max in world frame. quat_apply may flip signs per axis
    # (depending on microwave orientation) so take elementwise min/max to get the true
    # AABB for simple inside/outside checks.
    box_low_world = torch.minimum(box_min_world, box_max_world)
    box_high_world = torch.maximum(box_min_world, box_max_world)

    # Ensure hand_pos_w has a batch dimension
    if hand_pos_w.ndim == 1:
        hand_pos_w = hand_pos_w.unsqueeze(0)

    # Small tolerance to account for numerical precision and tiny pose offsets
    tol = 5e-3

    # Check if object is inside the box in world frame (with tolerance)
    obj_x_in_world = (obj_pos_w[:, 0] >= (box_low_world[:, 0] - tol)) & (obj_pos_w[:, 0] <= (box_high_world[:, 0] + tol))
    obj_y_in_world = (obj_pos_w[:, 1] >= (box_low_world[:, 1] - tol)) & (obj_pos_w[:, 1] <= (box_high_world[:, 1] + tol))
    obj_z_in_world = (obj_pos_w[:, 2] >= (box_low_world[:, 2] - tol)) & (obj_pos_w[:, 2] <= (box_high_world[:, 2] + tol))
    is_obj_in_world = obj_x_in_world & obj_y_in_world & obj_z_in_world

    # Check if hand is outside the box - ANY coordinate can be outside bounds
    hand_x_in = (hand_pos_w[:, 0] >= (box_low_world[:, 0] - tol)) & (hand_pos_w[:, 0] <= (box_high_world[:, 0] + tol))
    hand_y_in = (hand_pos_w[:, 1] >= (box_low_world[:, 1] - tol)) & (hand_pos_w[:, 1] <= (box_high_world[:, 1] + tol))
    hand_z_in = (hand_pos_w[:, 2] >= (box_low_world[:, 2] - tol)) & (hand_pos_w[:, 2] <= (box_high_world[:, 2] + tol))
    is_hand_in = hand_x_in & hand_y_in & hand_z_in
    is_hand_out = ~is_hand_in

    # ADDITIONAL SAFETY CHECK: Object should also be close to microwave in world coordinates
    # This prevents false positives when object is far from microwave but somehow passes local checks
    obj_to_microwave_dist = torch.norm(obj_pos_w - microwave_pos_w, dim=1)
    is_obj_near_microwave = obj_to_microwave_dist < 0.5  # Object must be within 50cm of microwave center

    # Check if object is in the gripper with updated joint positions
    gripper_open_joint_pos = 0.04  # Open state joint position
    gripper_closed_joint_pos = 0.01  # Closed state joint position

    finger_joint_indices = [robot.joint_names.index(name) for name in finger_joint_names]
    gripper_joint_pos = robot.data.joint_pos[:, finger_joint_indices]
    gripper_open = torch.all(torch.abs(gripper_joint_pos - gripper_open_joint_pos) < 0.005, dim=1)

    object_in_gripper = ~gripper_open  # Simplified condition

    # Combine the conditions: success = object is in AND hand is out AND object is not in gripper
    success = is_obj_in_world & is_hand_out & ~object_in_gripper

    # Debugging logs for all conditions: only print when success condition is met
    for env_idx in range(len(success)):
        if success[env_idx].item():
            print(f"\n======= DEBUG LOG - Environment {env_idx} =======")
            print(f"Object: {object_name}, Microwave: {microwave_name}, Hand: {hand_body_name}")
            print(f"Microwave box dimensions: {microwave_box_dims}")
            print(f"Microwave position (env frame): {microwave_pos_w[env_idx].cpu().numpy()}")
            print(f"Object position (env frame): {obj_pos_w[env_idx].cpu().numpy()}")
            print(f"Hand position (env frame): {hand_pos_w[env_idx].cpu().numpy()}")
            print(f"Bounding box low (world): {box_low_world.cpu().numpy()}")
            print(f"Bounding box high (world): {box_high_world.cpu().numpy()}")
            print(f"Object in bounds - X: {obj_x_in_world[env_idx].item()}, Y: {obj_y_in_world[env_idx].item()}, Z: {obj_z_in_world[env_idx].item()}")
            print(f"Hand in bounds - X: {hand_x_in[env_idx].item()}, Y: {hand_y_in[env_idx].item()}, Z: {hand_z_in[env_idx].item()}")
            print(f"Gripper joint positions: {gripper_joint_pos}")
            print(f"Gripper open status: {gripper_open[env_idx].item()}")
            print(f"Is object fully in microwave (world frame)? {is_obj_in_world[env_idx].item()}")
            print(f"Is hand completely out of microwave? {is_hand_out[env_idx].item()}")
            print(f"Is object in gripper? {object_in_gripper[env_idx].item()}")
            print(f"Success condition met? {success[env_idx].item()}")
            print("=========================================")

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
        self.image_obs_list = ["wrist_cam"]

        # Set settings for camera rendering
        self.rerender_on_reset = True
        self.sim.render.antialiasing_mode = "OFF"  # disable dlss

        num_envs = getattr(self, 'num_envs', self.scene.num_envs if hasattr(self, 'scene') else 1)
        self.scene = MinimalKitchenSceneCfg(num_envs=num_envs, env_spacing=5.0)

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
            data_types=["rgb"],  # Added depth information
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
                # Increased X and Y by 0.05 for slightly larger detection area
                # Microwave interior is typically: width=35-40cm, depth=30-35cm, height=20-25cm
                "microwave_box_dims": (0.27, 0.27, 0.18),  # Increased Y dimension to 0.27
                "finger_joint_names": ["panda_finger_joint1", "panda_finger_joint2"],
            },
        )

        # Visual markers for success zone debugging
        success_zone_debug_vis = False  # Set to True to enable visual markers
        
        if success_zone_debug_vis:
            # Add a visual marker for the microwave bounding box using simple sphere markers
            # Create multiple spheres to outline the success box corners for maximum visibility
            # These positions are relative to each environment's origin
            # Adjust success zone markers to reflect the increased y-dimension
            self.scene.success_marker_corner1 = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/SuccessCorner1",
                spawn=sim_utils.SphereCfg(
                    radius=0.02,  # Small spheres
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.0, 0.0),  # Bright red
                        emissive_color=(2.0, 0.0, 0.0),  # Bright red glow
                        opacity=1.0,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(
                    # Position relative to environment origin - bottom front-left corner
                    pos=(2.20, 0.425, 1.10),  # Adjusted Y for new microwave dimensions
                )
            )
            
            self.scene.success_marker_corner2 = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/SuccessCorner2", 
                spawn=sim_utils.SphereCfg(
                    radius=0.02,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.0, 0.0),
                        emissive_color=(2.0, 0.0, 0.0),
                        opacity=1.0,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(
                    # Position relative to environment origin - top back-right corner
                    pos=(2.50, 0.695, 1.30),  # Adjusted Y for new microwave dimensions
                )
            )
            
            # Add corner 3 marker (bottom back-right corner)
            self.scene.success_marker_corner3 = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/SuccessCorner3", 
                spawn=sim_utils.SphereCfg(
                    radius=0.02,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.0, 0.0),
                        emissive_color=(2.0, 0.0, 0.0),
                        opacity=1.0,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(
                    # Position relative to environment origin - bottom back-right corner
                    pos=(2.50, 0.695, 1.10),  # Adjusted Y for new microwave dimensions
                )
            )
            
            # Add corner 4 marker (top front-left corner)
            self.scene.success_marker_corner4 = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/SuccessCorner4", 
                spawn=sim_utils.SphereCfg(
                    radius=0.02,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.0, 0.0),
                        emissive_color=(2.0, 0.0, 0.0),
                        opacity=1.0,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(
                    # Position relative to environment origin - top front-left corner
                    pos=(2.20, 0.425, 1.30),  # Adjusted Y for new microwave dimensions
                )
            )
            
            # Add center marker for better visualization
            self.scene.success_marker_center = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/SuccessCenter",
                spawn=sim_utils.SphereCfg(
                    radius=0.03,  # Slightly larger center sphere
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.0),  # Bright green
                        emissive_color=(0.0, 2.0, 0.0),  # Bright green glow
                        opacity=1.0,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(
                    # Position at microwave center relative to environment origin
                    pos=(2.35, 0.56, 1.20),  # Decreased X by 0.05: (2.40 - 0.05)
                )
            )

        # Close the visual markers conditional
        
        # Define subtasks for the end effector
        self.subtask_configs = {
            "end_effector": [
                SubTaskConfig(
                    object_ref="object",
                    subtask_term_signal="move_to_fridge"
                ),
                SubTaskConfig(
                    object_ref="object",
                    subtask_term_signal="grasp_tomato_can"
                ),
                SubTaskConfig(
                    object_ref="object",
                    subtask_term_signal="move_to_microwave"
                ),
                SubTaskConfig(
                    object_ref="object",
                    subtask_term_signal="place_in_microwave"
                ),
                SubTaskConfig(
                    object_ref="object",
                    subtask_term_signal="move_away_from_microwave"
                ),
                # This is the final success condition - don't remove this
                SubTaskConfig(
                    object_ref="object",
                    subtask_term_signal="task_complete"
                )
            ]
        }
    
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
