# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, mdp
from .cognitive_robotics_genreal_env_cfg import CognitiveRoboticsGenrealEnvCfg, CognitiveRoboticsGenrealEnvCfg_PLAY

##
# Register Gym environments.
##

gym.register(
    id="CognitiveRoboticsGenreal-Lift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "cognitive_robotics_genreal.tasks.manager_based.cognitive_robotics_genreal:CognitiveRoboticsGenrealEnvCfg",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.LiftPPORunnerCfg,
        "skrl_cfg_entry_point": "cognitive_robotics_genreal.tasks.manager_based.cognitive_robotics_genreal.agents:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="CognitiveRoboticsGenreal-Lift-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "cognitive_robotics_genreal.tasks.manager_based.cognitive_robotics_genreal:CognitiveRoboticsGenrealEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.LiftPPORunnerCfg,
        "skrl_cfg_entry_point": "cognitive_robotics_genreal.tasks.manager_based.cognitive_robotics_genreal.agents:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)
