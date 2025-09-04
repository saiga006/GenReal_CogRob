# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom event functions for the lift environment."""

import torch
import omni
from pxr import UsdLux


def randomize_light_intensity(env, env_ids: torch.Tensor, light_path: str = "/World/light", intensity_range: tuple = (500.0, 1500.0)):
    """
    Randomizes the light intensity of a dome light in the scene.
    
    Args:
        env: The environment instance.
        env_ids: The environment indices to apply the randomization to.
        light_path: The prim path to the light.
        intensity_range: The range (min, max) of intensity values to sample from.
    """
    # Get number of environments to randomize
    num_envs_to_randomize = len(env_ids)
    
    # Generate random intensity values for the specified environments
    min_intensity, max_intensity = intensity_range
    random_intensities = torch.rand(num_envs_to_randomize, device=env.device) * (max_intensity - min_intensity) + min_intensity
    
    # Get stage
    stage = omni.usd.get_context().get_stage()
    
    # For each environment that needs randomization
    for i, env_idx in enumerate(env_ids):
        env_idx = int(env_idx.item())  # Convert to int
        
        # Get light path for this environment
        env_light_path = light_path
        if "{ENV_REGEX_NS}" in light_path:
            env_light_path = light_path.replace("{ENV_REGEX_NS}", f"/World/envs/env_{env_idx}")
        
        # Get the light prim
        light_prim = stage.GetPrimAtPath(env_light_path)
        
        if light_prim.IsValid():
            # Try different possible attribute names for intensity
            intensity_attr_names = ["inputs:intensity", "intensity", "inputs:exposure", "exposure"]
            intensity_attr = None
            
            for attr_name in intensity_attr_names:
                test_attr = light_prim.GetAttribute(attr_name)
                if test_attr.IsValid():
                    intensity_attr = test_attr
                    break
            
            if intensity_attr and intensity_attr.IsValid():
                new_intensity = float(random_intensities[i].item())
                intensity_attr.Set(new_intensity)
                print(f"Light intensity randomized to {new_intensity:.1f} for environment {env_idx}")
            else:
                print(f"Warning: Could not find intensity attribute for light at {env_light_path}")
        else:
            print(f"Warning: Light prim not found at path {env_light_path}")
    
    return torch.ones((len(env_ids), 1), dtype=torch.bool, device=env.device)
