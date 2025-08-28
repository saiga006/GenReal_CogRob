#!/usr/bin/env python3

"""Launch kitchen teleoperation with proper environment registration."""

from isaaclab.app import AppLauncher

# Launch the simulation app
app_launcher = AppLauncher()
simulation_app = app_launcher.app

import os
import sys
import argparse

# Add kitchen_env to Python path
kitchen_env_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, kitchen_env_path)

# Import to register the environment BEFORE any gym operations
import envs

# Now import Isaac Lab modules
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.envs import ManagerBasedRLEnv
from scripts.environments.teleoperation.teleop_se3_agent import TeleopSE3Agent

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Teleoperation for kitchen environment")
    parser.add_argument("--task", type=str, default="Isaac-Kitchen-Teleop-Kinova-v0", help="Name of the task.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the simulation on.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
    parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for teleoperation.")
    args = parser.parse_args()

    # Parse configuration
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    
    # Create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # Create teleoperation interface
    teleop_interface = TeleopSE3Agent(cfg=env_cfg, env=env)
    
    # Run teleoperation
    teleop_interface.run()
    
    # Close the simulation
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
