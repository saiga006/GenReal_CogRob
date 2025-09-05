# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def ee_fridge_distance(
    env: ManagerBasedRLEnv,
    std: float,
    fridge_cfg: SceneEntityCfg = SceneEntityCfg("fridge_base"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the fridge using tanh-kernel."""
    fridge: RigidObject = env.scene[fridge_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    fridge_pos_w = fridge.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]

    distance = torch.norm(fridge_pos_w - ee_w, dim=1)
    return 1 - torch.tanh(distance / std)


def object_grasped(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for grasping the object."""
    obj: RigidObject = env.scene[object_cfg.name]
    # Assuming the gripper is closed when the object is lifted
    return torch.where(obj.data.root_pos_w[:, 2] > 0.9, 1.0, 0.0)


def ee_microwave_distance(
    env: ManagerBasedRLEnv,
    std: float,
    microwave_cfg: SceneEntityCfg = SceneEntityCfg("microwave"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the microwave using tanh-kernel."""
    microwave: Articulation = env.scene[microwave_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    microwave_pos_w = microwave.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]

    distance = torch.norm(microwave_pos_w - ee_w, dim=1)
    return 1 - torch.tanh(distance / std)


def object_in_microwave(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    microwave_cfg: SceneEntityCfg = SceneEntityCfg("microwave"),
) -> torch.Tensor:
    """Reward for placing the object inside the microwave."""
    obj: RigidObject = env.scene[object_cfg.name]
    microwave: Articulation = env.scene[microwave_cfg.name]

    microwave_pos_w = microwave.data.root_pos_w
    obj_pos_w = obj.data.root_pos_w

    # Simple check if object is within a bounding box relative to microwave
    # This might need tuning based on the actual microwave model
    local_pos = obj_pos_w - microwave_pos_w
    in_microwave = (torch.abs(local_pos[:, 0]) < 0.2) & \
                   (torch.abs(local_pos[:, 1]) < 0.2) & \
                   (local_pos[:, 2] > 0.0) & (local_pos[:, 2] < 0.3)

    return torch.where(in_microwave, 1.0, 0.0)
