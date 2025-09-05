#!/usr/bin/env python3
"""
Teleoperation Data Collection for Kinova Gen3 Kitchen Manipulation

This script allows manual control of the Kinova arm through keyboard input
and logs demonstration trajectories for imitation learning.

Controls:
- w/s: Move end-effector forward/back (X-axis)
- a/d: Move end-effector left/right (Y-axis)  
- q/e: Move end-effector up/down (Z-axis)
- i/k: Pitch rotation (up/down)
- j/l: Yaw rotation (left/right)
- u/o: Roll rotation
- g: Close gripper
- h: Open gripper
- space: Reset to home position
- enter: End current episode
- esc: Quit program

Usage:
    python3 scripts/collect_teleop.py [options]

Example:
    python3 scripts/collect_teleop.py --episodes 10 --save-images
"""

import os
import sys
import time
import argparse
import threading
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import numpy as np
import yaml

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import mujoco
import mujoco.viewer
from utils.logger import TrajectoryLogger

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("⚠️  pygame not available. Install with: pip install pygame")

try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("⚠️  pynput not available. Install with: pip install pynput")


class TeleopController:
    """Teleoperation controller for the Kinova Gen3 arm."""
    
    def __init__(self, scene_xml: str, config: Dict[str, Any], headless: bool = False):
        """
        Initialize the teleoperation controller.
        
        Args:
            scene_xml: Path to the MuJoCo XML scene file
            config: Configuration dictionary
            headless: Whether to run without visual display
        """
        self.config = config
        self.headless = headless
        self.running = True
        self.episode_active = False
        self.current_episode = 0
        
        # Control state
        self.velocity_command = np.zeros(6)  # [x, y, z, rx, ry, rz] end-effector velocity
        self.gripper_command = 0.0  # -1: open, 1: close, 0: hold
        self.reset_requested = False
        self.end_episode_requested = False
        
        # Control parameters
        self.linear_speed = 0.1   # m/s
        self.angular_speed = 0.5  # rad/s
        self.gripper_speed = 0.02
        self.control_freq = 20    # Hz
        
        print(f"🏠 Loading kitchen scene: {scene_xml}")
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)
        
        # Initialize viewer
        self.viewer = None
        if not headless:
            print("📺 Launching MuJoCo viewer...")
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            print("✅ Visual simulation window opened!")
            print("🎮 Use keyboard controls to manipulate the robot")
        
        # Initialize home position (safe starting pose)
        self.home_position = np.array([0.0, -0.5, 0.0, -1.2, 0.0, 0.8, 0.0])
        
        # Get end-effector site ID for Cartesian control
        try:
            self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        except:
            # Fallback: create a virtual end-effector position
            self.ee_site_id = None
            print("⚠️  No 'attachment_site' found, using last link position")
        
        # Initialize trajectory logger
        self.trajectory_logger = None
        self._init_logger()
        
        # Initialize input handler
        self._init_input_handler()
        
        print("✅ TeleopController initialized successfully!")
        self._print_controls()
    
    def _init_logger(self):
        """Initialize trajectory logging."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"dataset/teleop_{timestamp}"
        
        logger_config = {
            'save_images': self.config.get('logging', {}).get('save_images', False),
            'states_only': self.config.get('logging', {}).get('states_only', True),
            'format': 'jsonl'
        }
        
        print("📊 Initializing trajectory logger...")
        self.trajectory_logger = TrajectoryLogger(log_dir, **logger_config)
    
    def _init_input_handler(self):
        """Initialize keyboard input handling."""
        if PYGAME_AVAILABLE:
            self._init_pygame()
        elif PYNPUT_AVAILABLE:
            self._init_pynput()
        else:
            raise RuntimeError("Neither pygame nor pynput available. Install one with: pip install pygame")
    
    def _init_pygame(self):
        """Initialize pygame for keyboard input."""
        pygame.init()
        self.screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption("Kinova Teleop Controls")
        self.clock = pygame.time.Clock()
        print("🎮 Using pygame for keyboard input")
    
    def _init_pynput(self):
        """Initialize pynput for keyboard input."""
        self.pressed_keys = set()
        
        def on_press(key):
            try:
                self.pressed_keys.add(key.char)
            except AttributeError:
                # Special keys
                if key == keyboard.Key.space:
                    self.pressed_keys.add('space')
                elif key == keyboard.Key.enter:
                    self.pressed_keys.add('enter')
                elif key == keyboard.Key.esc:
                    self.pressed_keys.add('esc')
        
        def on_release(key):
            try:
                self.pressed_keys.discard(key.char)
            except AttributeError:
                if key == keyboard.Key.space:
                    self.pressed_keys.discard('space')
                elif key == keyboard.Key.enter:
                    self.pressed_keys.discard('enter')
                elif key == keyboard.Key.esc:
                    self.pressed_keys.discard('esc')
        
        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()
        print("🎮 Using pynput for keyboard input")
    
    def _print_controls(self):
        """Print control instructions."""
        print("\n" + "="*60)
        print("🎮 TELEOPERATION CONTROLS")
        print("="*60)
        print("Movement:")
        print("  w/s  - Forward/Back (X-axis)")
        print("  a/d  - Left/Right (Y-axis)")
        print("  q/e  - Up/Down (Z-axis)")
        print("\nRotation:")
        print("  i/k  - Pitch up/down")
        print("  j/l  - Yaw left/right")
        print("  u/o  - Roll")
        print("\nGripper:")
        print("  g    - Close gripper")
        print("  h    - Open gripper")
        print("\nEpisode Control:")
        print("  space - Reset to home position")
        print("  enter - End current episode")
        print("  esc   - Quit program")
        print("="*60)
        print("💡 Focus the pygame window or console for keyboard input")
        print("🚀 Press ENTER to start first episode...")
        print()
    
    def _get_current_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current end-effector position and orientation."""
        if self.ee_site_id is not None:
            # Use the attachment site
            pos = self.data.site_xpos[self.ee_site_id].copy()
            mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
        else:
            # Use the last body (end-effector link)
            pos = self.data.xpos[-1].copy()  # Last body position
            mat = self.data.xmat[-1].reshape(3, 3)  # Last body orientation
        
        # Convert rotation matrix to euler angles (simplified)
        # This is a basic implementation - for production use scipy.spatial.transform
        euler = np.array([
            np.arctan2(mat[2, 1], mat[2, 2]),  # Roll
            np.arctan2(-mat[2, 0], np.sqrt(mat[2, 1]**2 + mat[2, 2]**2)),  # Pitch
            np.arctan2(mat[1, 0], mat[0, 0])   # Yaw
        ])
        
        return pos, euler
    
    def _update_velocity_from_input(self):
        """Update velocity command based on current input."""
        vel = np.zeros(6)
        
        if PYGAME_AVAILABLE:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        self.end_episode_requested = True
                    elif event.key == pygame.K_SPACE:
                        self.reset_requested = True
                    elif event.key == pygame.K_ESCAPE:
                        self.running = False
            
            # Get current key states
            keys = pygame.key.get_pressed()
            
            # Linear movement
            if keys[pygame.K_w]: vel[0] += self.linear_speed    # Forward
            if keys[pygame.K_s]: vel[0] -= self.linear_speed    # Back
            if keys[pygame.K_d]: vel[1] += self.linear_speed    # Right  
            if keys[pygame.K_a]: vel[1] -= self.linear_speed    # Left
            if keys[pygame.K_e]: vel[2] += self.linear_speed    # Up
            if keys[pygame.K_q]: vel[2] -= self.linear_speed    # Down
            
            # Angular movement
            if keys[pygame.K_i]: vel[3] += self.angular_speed   # Pitch up
            if keys[pygame.K_k]: vel[3] -= self.angular_speed   # Pitch down
            if keys[pygame.K_l]: vel[4] += self.angular_speed   # Yaw right
            if keys[pygame.K_j]: vel[4] -= self.angular_speed   # Yaw left
            if keys[pygame.K_u]: vel[5] += self.angular_speed   # Roll left
            if keys[pygame.K_o]: vel[5] -= self.angular_speed   # Roll right
            
            # Gripper
            self.gripper_command = 0.0
            if keys[pygame.K_g]: self.gripper_command = 1.0     # Close
            if keys[pygame.K_h]: self.gripper_command = -1.0    # Open
            
        elif PYNPUT_AVAILABLE:
            # Handle pynput input
            if 'esc' in self.pressed_keys:
                self.running = False
            if 'enter' in self.pressed_keys:
                self.end_episode_requested = True
                self.pressed_keys.discard('enter')  # Consume the event
            if 'space' in self.pressed_keys:
                self.reset_requested = True
                self.pressed_keys.discard('space')  # Consume the event
            
            # Linear movement
            if 'w' in self.pressed_keys: vel[0] += self.linear_speed
            if 's' in self.pressed_keys: vel[0] -= self.linear_speed
            if 'd' in self.pressed_keys: vel[1] += self.linear_speed
            if 'a' in self.pressed_keys: vel[1] -= self.linear_speed
            if 'e' in self.pressed_keys: vel[2] += self.linear_speed
            if 'q' in self.pressed_keys: vel[2] -= self.linear_speed
            
            # Angular movement
            if 'i' in self.pressed_keys: vel[3] += self.angular_speed
            if 'k' in self.pressed_keys: vel[3] -= self.angular_speed
            if 'l' in self.pressed_keys: vel[4] += self.angular_speed
            if 'j' in self.pressed_keys: vel[4] -= self.angular_speed
            if 'u' in self.pressed_keys: vel[5] += self.angular_speed
            if 'o' in self.pressed_keys: vel[5] -= self.angular_speed
            
            # Gripper
            self.gripper_command = 0.0
            if 'g' in self.pressed_keys: self.gripper_command = 1.0
            if 'h' in self.pressed_keys: self.gripper_command = -1.0
        
        self.velocity_command = vel
    
    def _velocity_to_joint_space(self, ee_velocity: np.ndarray) -> np.ndarray:
        """
        Convert end-effector velocity to joint velocities using Jacobian.
        
        Args:
            ee_velocity: [x, y, z, rx, ry, rz] end-effector velocity
            
        Returns:
            Joint velocity commands
        """
        # Get current end-effector position for Jacobian calculation
        pos, _ = self._get_current_ee_pose()
        
        # Compute Jacobian (6x7 for 7-DOF arm)
        jac_pos = np.zeros((3, self.model.nv))  # Position Jacobian
        jac_rot = np.zeros((3, self.model.nv))  # Rotation Jacobian
        
        if self.ee_site_id is not None:
            mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, self.ee_site_id)
        else:
            # Use the last body
            body_id = self.model.nbody - 1
            mujoco.mj_jacBody(self.model, self.data, jac_pos, jac_rot, body_id)
        
        # Combine position and rotation Jacobians
        jacobian = np.vstack([jac_pos, jac_rot])
        
        # Only use arm joints (first 7 DOF)
        jacobian = jacobian[:, :7]
        
        # Pseudo-inverse for velocity control
        try:
            jac_pinv = np.linalg.pinv(jacobian)
            joint_vel = jac_pinv @ ee_velocity
        except np.linalg.LinAlgError:
            # Fallback: zero velocity if Jacobian is singular
            joint_vel = np.zeros(7)
        
        return joint_vel
    
    def _apply_control(self):
        """Apply current control commands to the robot."""
        if self.reset_requested:
            # Reset to home position
            self.data.qpos[:7] = self.home_position
            self.data.qvel[:7] = 0.0
            mujoco.mj_forward(self.model, self.data)
            self.reset_requested = False
            print("🏠 Reset to home position")
            return
        
        # Convert end-effector velocity to joint space
        if np.any(self.velocity_command != 0):
            joint_vel = self._velocity_to_joint_space(self.velocity_command)
            
            # Apply joint velocities as position targets (integration)
            dt = 1.0 / self.control_freq
            current_pos = self.data.qpos[:7].copy()
            target_pos = current_pos + joint_vel * dt
            
            # Clip to joint limits
            for i in range(7):
                joint_range = self.model.jnt_range[i]
                if joint_range[0] < joint_range[1]:  # Valid range
                    target_pos[i] = np.clip(target_pos[i], joint_range[0], joint_range[1])
            
            # Send position commands to actuators
            self.data.ctrl[:7] = target_pos
        
        # Apply gripper control (if gripper actuators exist)
        if len(self.data.ctrl) > 7 and self.gripper_command != 0:
            # Simple gripper control - adjust as needed for your gripper
            gripper_pos = self.data.ctrl[7:] if len(self.data.ctrl) > 7 else []
            for i in range(len(gripper_pos)):
                gripper_pos[i] += self.gripper_command * self.gripper_speed
                gripper_pos[i] = np.clip(gripper_pos[i], -1.0, 1.0)
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation."""
        ee_pos, ee_euler = self._get_current_ee_pose()
        
        return {
            'joint_positions': self.data.qpos[:7].copy(),
            'joint_velocities': self.data.qvel[:7].copy(),
            'ee_position': ee_pos,
            'ee_orientation': ee_euler,
            'gripper_pos': self.data.qpos[7:].copy() if self.model.nq > 7 else np.array([]),
            'timestamp': time.time()
        }
    
    def _get_action(self) -> np.ndarray:
        """Get current action (control inputs)."""
        return self.data.ctrl.copy()
    
    def _start_episode(self) -> str:
        """Start a new episode."""
        self.current_episode += 1
        self.episode_active = True
        self.end_episode_requested = False
        
        # Reset to home position
        self.data.qpos[:7] = self.home_position
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        
        # Start logging
        episode_id = f"traj_{self.current_episode:04d}"
        self.trajectory_logger.start_episode(episode_id)
        
        print(f"🎬 Episode {self.current_episode} started - Use keyboard to control robot")
        return episode_id
    
    def _end_episode(self) -> Dict[str, Any]:
        """End current episode."""
        if not self.episode_active:
            return {}
        
        self.episode_active = False
        
        # End logging and get metadata
        metadata = self.trajectory_logger.end_episode()
        metadata.update({
            'episode_number': self.current_episode,
            'control_method': 'teleoperation',
            'operator': 'human',
            'success': True,  # Could add success detection logic
            'task_completion': 1.0  # Could add completion percentage
        })
        
        print(f"📊 Episode {self.current_episode} completed")
        return metadata
    
    def run_episode(self) -> Dict[str, Any]:
        """Run a single teleoperation episode."""
        # Wait for user to start episode
        if not self.episode_active:
            while self.running and not self.end_episode_requested:
                self._update_velocity_from_input()
                if PYGAME_AVAILABLE:
                    pygame.display.flip()
                    self.clock.tick(self.control_freq)
                else:
                    time.sleep(1.0 / self.control_freq)
            
            if not self.running:
                return {}
            
            # Start episode
            episode_id = self._start_episode()
        
        # Main control loop
        step_count = 0
        while self.running and self.episode_active and not self.end_episode_requested:
            # Update input
            self._update_velocity_from_input()
            
            # Apply controls
            self._apply_control()
            
            # Step simulation
            mujoco.mj_step(self.model, self.data)
            
            # Log data
            if self.episode_active:
                obs = self._get_observation()
                action = self._get_action()
                info = {
                    'step': step_count,
                    'ee_velocity_command': self.velocity_command.copy(),
                    'gripper_command': self.gripper_command
                }
                
                self.trajectory_logger.log_step(obs, action, 0.0, info)  # reward=0 for teleop
                step_count += 1
            
            # Update viewer
            if self.viewer is not None:
                self.viewer.sync()
            
            # Control timing
            if PYGAME_AVAILABLE:
                pygame.display.flip()
                self.clock.tick(self.control_freq)
            else:
                time.sleep(1.0 / self.control_freq)
        
        # End episode
        return self._end_episode()
    
    def close(self):
        """Clean up resources."""
        print("🧹 Cleaning up...")
        
        if hasattr(self, 'listener') and self.listener:
            self.listener.stop()
        
        if PYGAME_AVAILABLE:
            pygame.quit()
        
        if self.viewer is not None:
            self.viewer.close()
        
        if self.trajectory_logger:
            self.trajectory_logger.close()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"⚠️  Config file not found: {config_path}")
        return {
            'task': {'name': 'kitchen_teleop'},
            'simulation': {'timestep': 0.002},
            'logging': {'save_images': False, 'states_only': True}
        }


def main():
    """Main teleoperation data collection."""
    parser = argparse.ArgumentParser(description='Kinova Gen3 Teleoperation Data Collection')
    parser.add_argument('--config', default='config/scene.yaml',
                      help='Configuration file path')
    parser.add_argument('--scene', default='custom_kitchen.xml',
                      help='MuJoCo scene XML file')
    parser.add_argument('--episodes', type=int, default=5,
                      help='Number of episodes to collect')
    parser.add_argument('--headless', action='store_true',
                      help='Run without visual display')
    parser.add_argument('--save-images', action='store_true',
                      help='Save camera images during episodes')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    if args.save_images:
        config['logging']['save_images'] = True
    
    print("🤖 Kinova Gen3 Teleoperation Data Collection")
    print("="*60)
    print(f"📁 Config: {args.config}")
    print(f"🏠 Scene: {args.scene}")
    print(f"📊 Episodes: {args.episodes}")
    print(f"📺 Visual: {not args.headless}")
    print("="*60)
    
    # Initialize controller
    controller = None
    try:
        controller = TeleopController(args.scene, config, args.headless)
        
        # Collect episodes
        successful_episodes = 0
        total_steps = 0
        
        for episode in range(args.episodes):
            if not controller.running:
                break
            
            print(f"\n🎬 Preparing Episode {episode + 1}/{args.episodes}")
            metadata = controller.run_episode()
            
            if metadata:
                successful_episodes += 1
                episode_steps = metadata.get('num_steps', 0)
                total_steps += episode_steps
                print(f"✅ Episode {episode + 1}: {episode_steps} steps")
            
            if not controller.running:
                break
        
        # Summary
        print("\n" + "="*60)
        print("📊 COLLECTION SUMMARY")
        print("="*60)
        print(f"Episodes collected: {successful_episodes}/{args.episodes}")
        print(f"Total steps: {total_steps}")
        print(f"Average steps/episode: {total_steps/max(successful_episodes, 1):.1f}")
        print(f"Dataset location: {controller.trajectory_logger.log_dir}")
        print("✅ Teleoperation data collection completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Collection interrupted by user")
    except Exception as e:
        print(f"❌ Error during collection: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if controller:
            controller.close()


if __name__ == "__main__":
    main()
