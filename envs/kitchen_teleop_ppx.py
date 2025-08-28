import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .kitchen_scene import MinimalKitchenEnvCfg

@configclass
class KitchenTeleopTaskCfg(ManagerBasedRLEnvCfg):
    """Configuration for kitchen teleoperation task with Kinova Gen3."""
    
    # Scene settings
    scene = MinimalKitchenEnvCfg(ManagerBasedRLEnvCfg).scene
    
    # Basic settings
    decimation = 2
    episode_length_s = 120.0
    
    # Observation space (minimal for teleoperation)
    observations = ObsGroup({
        "policy": ObsGroup({
            # Robot joint positions and velocities
            "joint_pos": ObsTerm(func="joint_pos_rel", params={"asset_cfg": SceneEntityCfg("robot")}),
            "joint_vel": ObsTerm(func="joint_vel_rel", params={"asset_cfg": SceneEntityCfg("robot")}),
            # End-effector pose relative to base
            "ee_pose": ObsTerm(
                func="ee_pose", 
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=["robotiq_85_left_finger_tip_link"]),
                }
            ),
            # Object pose (bowl)
            "object_pose": ObsTerm(
                func="object_pose", 
                params={"asset_cfg": SceneEntityCfg("bowl")}
            ),
        })
    })
    
    # Action space for SE(3) control + gripper
    actions = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,))  # 6DOF pose + gripper
    
    # Events for resetting
    events = {
        "reset_scene": EventTerm(func="reset_to_default", mode="reset"),
    }
