#!/usr/bin/env python3
"""Script with effort-based microwave control."""

import torch
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
    print("Creating minimal kitchen scene with effort-based microwave control...")
    
    # Initialize simulation
    sim_cfg = sim_utils.SimulationCfg(dt=1.0/60.0)
    sim = SimulationContext(sim_cfg)
    
    # Create minimal scene
    scene_cfg = MinimalKitchenSceneCfg(num_envs=1, env_spacing=5.0)
    scene = InteractiveScene(scene_cfg)
    
    # Reset simulation
    sim.reset()
    print("Simulation reset complete.")
    
    # Get microwave articulation
    microwave = scene["microwave"]
    
    print("=== MICROWAVE JOINT INFO ===")
    print(f"Joint names: {microwave.joint_names}")
    print(f"Joint limits: {microwave.data.joint_pos_limits}")
    print("============================")
    
    try:
        step_count = 0
        door_open_state = False
        
        while simulation_app.is_running():
            simulation_app.update()
            sim.step()
            
            try:
                scene.update(dt=sim.get_physics_dt())
            except:
                pass
            
            # Apply continuous effort to move joints
            if step_count % 600 == 0:
                door_open_state = not door_open_state
                
            # Apply effort based on desired state
            if door_open_state:
                # Apply negative effort to open door (towards negative limit)
                # Apply positive effort to rotate turntable
                efforts = torch.tensor([[-5.0, 2.0]], device=sim.device)  # [door_effort, turntable_effort]
                print("Applying effort to open door and rotate turntable...")
            else:
                # Apply positive effort to close door (towards zero)
                # Stop turntable
                efforts = torch.tensor([[2.0, 0.0]], device=sim.device)
                print("Applying effort to close door and stop turntable...")
            
            # Use effort control instead of position control
            microwave.set_joint_effort_target(efforts)
            microwave.write_data_to_sim()
            
            step_count += 1
            
            # Debug output
            if step_count % 300 == 0:
                try:
                    current_positions = microwave.data.joint_pos[0]
                    print(f"Step {step_count}: Door={current_positions[0]:.4f}, Turntable={current_positions[1]:.4f}")
                except:
                    print(f"Step {step_count}: Joint data not available")
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()
