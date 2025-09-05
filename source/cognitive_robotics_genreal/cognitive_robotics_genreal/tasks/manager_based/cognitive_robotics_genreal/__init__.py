# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

from .cognitive_robotics_genreal_env_cfg import CognitiveRoboticsGenrealEnvCfg, CognitiveRoboticsGenrealEnvCfg_PLAY

##
# Register Gym environments.
##

gym.register(
    id="Cognitive-Robotics-Genreal-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cognitive_robotics_genreal_env_cfg:CognitiveRoboticsGenrealEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

##
# Kitchen Scene with IK Control for Teleoperation (Relative Mode)
##

gym.register(
    id="Isaac-Kitchen-Kinova-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:KinovaTeleopEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-Kinova-IK-Rel-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:KinovaTeleopEnvCfg_PLAY",
    },
)