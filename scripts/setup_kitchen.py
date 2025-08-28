#!/usr/bin/env python3

"""Script to setup and visualize the kitchen environment."""

from isaaclab.app import AppLauncher

# Launch app
app_launcher = AppLauncher()
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

import sys
import os

# Import minimal scene
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from envs.kitchen_scene import MinimalKitchenSceneCfg


def main():
    print("Creating minimal kitchen scene...")
    
    # Initialize simulation
    sim_cfg = sim_utils.SimulationCfg(dt=1.0/60.0)
    sim = SimulationContext(sim_cfg)
    
    # Create minimal scene
    scene_cfg = MinimalKitchenSceneCfg(num_envs=1, env_spacing=5.0)
    scene = InteractiveScene(scene_cfg)
    # ... (after scene = InteractiveScene(scene_cfg))
    
    print("Minimal scene created! Should show a brown table.")
    print("Press Ctrl+C to exit.")
    
    try:
        step_count = 0
        while simulation_app.is_running():
            simulation_app.update()
            sim.step()
            
            try:
                scene.update(dt=sim.get_physics_dt())
            except:
                pass
            
            step_count += 1
            if step_count % 600 == 0:
                print(f"Running... (step {step_count})")
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()


