"""Register custom kitchen environments."""

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv

# Import your task configuration
from .kitchen_teleop_ppx import KitchenTeleopTaskCfg

# Register the task
gym.register(
    id="Isaac-Kitchen-Teleop-Kinova-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": KitchenTeleopTaskCfg,
    },
)

print("✓ Kitchen teleoperation environment registered: Isaac-Kitchen-Teleop-Kinova-v0")


