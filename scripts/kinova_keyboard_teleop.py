# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script provides a keyboard teleoperation interface for the Kinova Gen3 robot
in a kitchen environment, using differential inverse kinematics (IK) for control.

The control logic is adapted from the original Isaac Lab differential IK tutorial,
and the teleoperation interface is based on the `teleop_se3_agent.py` script.

Last modified: August 28, 2025
"""

# Standard library imports
import argparse
from copy import deepcopy

# Third-party imports
import torch
from isaaclab.app import AppLauncher

# Create argument parser to handle command line arguments
parser = argparse.ArgumentParser(description="Kinova Gen3 Keyboard Teleoperation using Differential IK.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to create.")
parser.add_argument("--pos_sensitivity", type=float, default=0.5, help="Teleoperation position sensitivity.")
parser.add_argument("--rot_sensitivity", type=float, default=0.5, help="Teleoperation rotation sensitivity.")
AppLauncher.add_app_launcher_args(parser)  # Add app-specific arguments
args_cli = parser.parse_args()

# Initialize application launcher
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app  # Get handle to the running application

# Import necessary modules after app initialization
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.devices import Se3Keyboard
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

from envs.kitchen_scene import MinimalKitchenSceneCfg


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, args):
    """Main simulation loop that handles robot control and visualization.

    Args:
        sim: Simulation context handle.
        scene: Initialized interactive scene.
        args: Command-line arguments.
    """
    # Extract robot entity from the scene
    robot = scene["robot"]

    # Initialize teleoperation interface
    teleop_interface = Se3Keyboard(
        pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity
    )
    print("Keyboard Teleoperation Interface:")
    print(teleop_interface)


    # Configure differential inverse kinematics controller
    # Use absolute mode for teleoperation (absolute commands)
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,  # Use absolute target poses
        ik_method="dls",
    )
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Configure visualization markers for end-effector and target
    ee_marker_cfg = deepcopy(FRAME_MARKER_CFG)
    ee_marker_cfg.prim_path = "/Visuals/ee_current"
    ee_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(ee_marker_cfg)

    goal_marker_cfg = deepcopy(FRAME_MARKER_CFG)
    goal_marker_cfg.prim_path = "/Visuals/ee_goal"
    goal_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    goal_marker = VisualizationMarkers(goal_marker_cfg)

    # Configure robot entity properties
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["joint_.*"], body_names=["end_effector_link"])
    robot_entity_cfg.resolve(scene)

    # Print joint and body information for verification
    print("Active joint names:", robot.data.joint_names)
    print("Active body names:", robot.data.body_names)

    # Helper functions for quaternion operations
    def quaternion_multiply(q1, q2):
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack([w, x, y, z], dim=-1)

    def quaternion_inverse(q):
        return torch.cat([q[..., 0:1], -q[..., 1:]], dim=-1)

    def axis_angle_to_quat(axis_angle):
        angle = torch.norm(axis_angle, dim=-1, keepdim=True)
        axis = axis_angle / (angle + 1e-8)
        cos_half = torch.cos(angle / 2)
        sin_half = torch.sin(angle / 2)
        return torch.cat([cos_half, sin_half * axis], dim=-1)

    def quat_to_axis_angle(quat):
        w = quat[..., 0]
        xyz = quat[..., 1:]
        angle = 2 * torch.acos(torch.clamp(w, -1, 1))
        sin_half = torch.sin(angle / 2)
        axis = xyz / (torch.norm(xyz, dim=-1, keepdim=True) + 1e-8)
        return axis * angle.unsqueeze(-1)

    def quat_rotate(q, v):
        """Rotates a vector v by a quaternion q."""
        q_w = q[..., 0]
        q_xyz = q[..., 1:]
        t = 2 * torch.cross(q_xyz, v)
        return v + q_w.unsqueeze(-1) * t + torch.cross(q_xyz, t)

    # Calculate end-effector Jacobian index
    # Find the end effector body index
    ee_body_name = "end_effector_link"
    ee_body_idx = robot.data.body_names.index(ee_body_name)
    if robot.is_fixed_base:
        ee_jacobi_idx = ee_body_idx - 1
    else:
        ee_jacobi_idx = ee_body_idx

    # Initialize simulation timing variables
    sim_dt = sim.get_physics_dt()
    count = 0

    # Reset robot to default state
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()
    # read the robot state
    scene.update(0.0)

    # Compute initial end-effector pose for relative mode
    ee_pose_w = robot.data.body_state_w[:, ee_body_idx, 0:7]
    root_pose_w = robot.data.root_state_w[:, 0:7]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )

    # Initialize controller with zero deltas (no movement initially)
    diff_ik_controller.reset()
    initial_command = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    diff_ik_controller.set_command(initial_command, ee_pos=ee_pos_b, ee_quat=ee_quat_b)

    # Initialize last target to current pose for holding
    last_target_pos = ee_pos_b.clone()
    last_target_quat = ee_quat_b.clone()

    # Set initial command for the controller
    ik_commands = torch.cat([last_target_pos, last_target_quat], dim=-1)
    diff_ik_controller.set_command(ik_commands)

    # Main simulation loop
    while simulation_app.is_running():
        # Get teleoperation command
        delta_pose, _ = teleop_interface.advance()
        
        # Compute current end-effector pose (needed for relative mode)
        ee_pose_w = robot.data.body_state_w[:, ee_body_idx, 0:7]
        root_pose_w = robot.data.root_state_w[:, 0:7]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        
        # In absolute mode, commands are absolute poses (7D: 3 pos + 4 quat)
        if delta_pose is not None:
            delta_tensor = torch.tensor(delta_pose, dtype=torch.float, device=robot.device).repeat(scene.num_envs, 1)
            delta_pos = delta_tensor[:, :3]
            delta_rot = delta_tensor[:, 3:]
            delta_quat = axis_angle_to_quat(delta_rot)

            # Update target position
            last_target_pos += quat_rotate(last_target_quat, delta_pos)
            
            # Update target orientation
            last_target_quat = quaternion_multiply(last_target_quat, delta_quat)

        # Set the absolute command for the IK controller
        ik_commands = torch.cat([last_target_pos, last_target_quat], dim=-1)
        diff_ik_controller.set_command(ik_commands)

        # Compute inverse kinematics
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
        joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        # Apply joint position targets
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)

        # Update simulation and visualization
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)

        # Update marker positions
        ee_pose_w = robot.data.body_state_w[:, ee_body_idx, 0:7]
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])

        # Visualize the goal pose
        target_pose_w_pos, target_pose_w_quat = combine_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ik_commands[:, 0:3], ik_commands[:, 3:7]
        )
        goal_marker.visualize(target_pose_w_pos, target_pose_w_quat)


def main():
    """Main function to configure and run the simulation."""
    # Configure simulation parameters
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)

    # Create minimal scene
    scene_cfg = MinimalKitchenSceneCfg(num_envs=args_cli.num_envs, env_spacing=5.0)
    scene = InteractiveScene(scene_cfg)
    
    print("Minimal scene created! Should show a brown table.")
    print("Press Ctrl+C to exit.")

    # Reset and run simulation
    sim.reset()
    run_simulator(sim, scene, args_cli)


if __name__ == "__main__":
    # Entry point for execution
    main()
    # Properly close the application
    simulation_app.close()
