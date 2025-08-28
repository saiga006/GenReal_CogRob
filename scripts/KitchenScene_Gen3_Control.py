# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Demonstration of differential inverse kinematics control for Kinova Gen3 manipulator.

This script implements end-effector pose control using differential inverse kinematics (Differential IK).
The implementation is adapted from the original tutorial:
[source/standalone/tutorials/05_controllers/run_diff_ik.py]

Last modified: January 23, 2025
"""

# Standard library imports
import argparse

# Third-party imports
import torch
from isaaclab.app import AppLauncher

# Create argument parser to handle command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to create")
AppLauncher.add_app_launcher_args(parser)  # Add app-specific arguments
args_cli = parser.parse_args()

# Initialize application launcher
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app  # Get handle to the running application

# Import necessary modules after app initialization
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

from envs.kitchen_scene import MinimalKitchenSceneCfg


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Main simulation loop that handles robot control and visualization.

    Args:
        sim: Simulation context handle
        scene: Initialized interactive scene
    """
    # Extract robot entity from the scene
    robot = scene["robot"]

    # Configure differential inverse kinematics controller
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",  # Control end-effector pose
        use_relative_mode=False,  # Use absolute target positions
        ik_method="dls"  # Damped least squares method for IK
    )
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Configure visualization markers for end-effector and target
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)  # Reduce marker size
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Define end-effector target poses [position(x,y,z) + quaternion(w,x,y,z)]
    # These are in the robot's base frame.
    ee_goals = [
        # Goal in front of the microwave
        [0.2, -0.6, 0.5, 0.5, 0.5, -0.5, -0.5],
        # Goal in front of the fridge - adjusted to avoid microwave
        [-0.4, -0.7, 0.5, 0.5, 0.5, -0.5, -0.5],
        # A goal in front of the robot - raised higher
        [0.4, 0.0, 0.7, 0.707, 0.0, 0.707, 0.0],
    ]

    ee_goals = torch.tensor(ee_goals, device=sim.device)
    current_goal_idx = 0  # Index to track current target
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = ee_goals[current_goal_idx]

    # Configure robot entity properties
    robot_entity_cfg = SceneEntityCfg(
        "robot",
        joint_names=["joint_.*"],          # Regex to match all arm joints (joint_1 to joint_7)
        body_names=["end_effector_link"]   # End-effector body name
    )
    robot_entity_cfg.resolve(scene)  # Resolve entity IDs from the scene

    # Print joint and body information for verification
    print("Active joint names:", robot.data.joint_names)
    print("Active body names:", robot.data.body_names)

    # Calculate end-effector Jacobian index based on robot configuration
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    # Initialize simulation timing variables
    sim_dt = sim.get_physics_dt()
    count = 0  # Step counter for target switching

    # Main simulation loop
    while simulation_app.is_running():
        if count % 300 == 0:
            # Reset robot state and switch target every 150 steps
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            
            # Update target command and reset controller
            ik_commands[:] = ee_goals[current_goal_idx]
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)
            
            # Cycle through targets
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
        else:
            # Compute inverse kinematics
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            root_pose_w = robot.data.root_state_w[:, 0:7]
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            
            # Calculate end-effector pose relative to robot base
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3],  # Base position
                root_pose_w[:, 3:7],  # Base orientation
                ee_pose_w[:, 0:3],    # End-effector position
                ee_pose_w[:, 3:7]     # End-effector orientation
            )
            
            # Compute desired joint positions using IK
            joint_pos_des = diff_ik_controller.compute(
                ee_pos_b, 
                ee_quat_b, 
                jacobian, 
                joint_pos
            )

        # Apply joint position targets
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        
        # Update simulation and visualization
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)

        # Update marker positions
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])

        # Transform goal from base frame to world frame for visualization
        root_pose_w = robot.data.root_state_w[:, 0:7]
        goal_pos_w, goal_rot_w = combine_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ik_commands[:, 0:3], ik_commands[:, 3:7]
        )
        goal_marker.visualize(goal_pos_w, goal_rot_w)


def main():
    """Main function to configure and run the simulation."""
    # Configure simulation parameters
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)  # 100Hz simulation
    sim = sim_utils.SimulationContext(sim_cfg)

    # Create minimal scene
    scene_cfg = MinimalKitchenSceneCfg(num_envs=1, env_spacing=5.0)
    scene = InteractiveScene(scene_cfg)
    # ... (after scene = InteractiveScene(scene_cfg))
    
    print("Minimal scene created! Should show a brown table.")
    print("Press Ctrl+C to exit.")

    # Reset and run simulation
    sim.reset()
    run_simulator(sim, scene)


if __name__ == "__main__":
    # Entry point for execution
    main()
    # Properly close the application
    simulation_app.close()