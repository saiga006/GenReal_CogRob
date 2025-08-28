#!/usr/bin/env python3

"""Simple teleoperation launcher."""

import os
import sys

# Add kitchen_env to path  
kitchen_env_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, kitchen_env_path)

# Import to register the environment
import envs

# Set command line arguments for the teleoperation script
sys.argv = [
    "teleop_se3_agent.py",
    "--task", "Isaac-Kitchen-Teleop-Kinova-v0", 
    "--num_envs", "1",
    "--teleop_device", "keyboard"
]

# Execute the teleoperation script directly
isaac_lab_root = "/home/saiga/Documents/Cognitive_Robotics/simulation_packages/IsaacLab"
teleop_script = os.path.join(isaac_lab_root, "scripts/environments/teleoperation/teleop_se3_agent.py")

with open(teleop_script, 'r') as f:
    exec(f.read())
