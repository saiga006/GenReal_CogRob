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
        light_path: The prim path to the light (should be global for shared lighting).
        intensity_range: The range (min, max) of intensity values to sample from.
    """
    # Generate a single random intensity value (since we're using global lighting)
    min_intensity, max_intensity = intensity_range
    random_intensity = torch.rand(1, device=env.device) * (max_intensity - min_intensity) + min_intensity
    
    # Get stage
    stage = omni.usd.get_context().get_stage()
    
    # Use the light path directly (should be global like "/World/light")
    light_prim = stage.GetPrimAtPath(light_path)
    
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
            new_intensity = float(random_intensity.item())
            intensity_attr.Set(new_intensity)
            # Only print if debug is needed - commented out to reduce console spam
            # print(f"Global light intensity randomized to {new_intensity:.1f}")
        else:
            print(f"Warning: Could not find intensity attribute for light at {light_path}")
    else:
        print(f"Warning: Light prim not found at path {light_path}")
    
    return torch.ones((len(env_ids), 1), dtype=torch.bool, device=env.device)
