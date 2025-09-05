#!/usr/bin/env python3
"""
Scripted Demo Collector for Kitchen Manipulation Task

Creates scripts/collect_scripted.py that:
- Loads the scene XML + config/scene.yaml
- Attaches a TrajectoryLogger
- Runs the behavior tree plan (fridge_to_microwave task)
- Records ~1 trajectory per run
- Loops for --episodes N (default 20)
- Resets environment at the start of each episode
- Calls logger.start_episode() and logger.end_episode(success=..., metadata={...})
- Metadata includes {"scripted": true, "task": "fridge_to_microwave"}
- Save files in dataset/YYYYMMDD_HHMMSS/traj_xxxx.jsonl
"""

import os
import sys
import argparse
import yaml
import mujoco
import mujoco.viewer
import numpy as np
import time
from datetime import datetime
from typing import Dict, Any, Optional

# Add project paths
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.logger import TrajectoryLogger
from skills.behavior_tree import KitchenBehaviorTree, TaskState


class ScriptedDemoCollector:
    """
    Collects demonstration trajectories using scripted behavior tree for kitchen manipulation.
    
    Task: Open fridge → pick object → place on microwave → close fridge → 
          open microwave → place inside → close microwave → press start
    """
    
    def __init__(self, config_path: str = "config/scene.yaml", 
                 scene_xml: str = "custom_kitchen.xml",
                 headless: bool = True):
        """Initialize the demo collector."""
        self.config_path = config_path
        self.scene_xml = scene_xml
        self.headless = headless
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize MuJoCo simulation
        print(f"Loading kitchen scene: {scene_xml}")
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)
        
        # Actuator diagnostics
        self._print_actuator_diagnostics()
        
        # Enable arm actuators and manage other actuators
        self._enable_actuators(include_prefixes=["j_", "arm_"])
        self._disable_non_arm_actuators()
        
        # Initialize viewer if not headless
        self.viewer = None
        if not headless:
            print("Launching MuJoCo viewer...")
            # Set graphics environment for proper display
            os.environ['DISPLAY'] = ':1'
            os.environ['MUJOCO_GL'] = 'glfw'
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            print("Visual simulation window opened!")
            print("Controls: Mouse to rotate, scroll to zoom, Tab for UI, Space to pause")
        
        # Initialize behavior tree
        print("Initializing behavior tree...")
        self.behavior_tree = KitchenBehaviorTree(self.config)
        
        # Initialize trajectory logger
        print("Initializing trajectory logger...")
        self.logger = TrajectoryLogger(
            log_dir="dataset/",
            format="jsonl",
            save_images=self.config.get('logging', {}).get('save_images', False),
            save_states_only=self.config.get('logging', {}).get('save_states_only', True)
        )
        
        print("ScriptedDemoCollector initialized successfully!")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load scene configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"Loaded config: {self.config_path}")
            return config
        except FileNotFoundError:
            print(f"Config file not found: {self.config_path}")
            return self._get_default_config()
        except yaml.YAMLError as e:
            print(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if YAML loading fails"""
        return {
            "task": {"name": "fridge_to_microwave"},
            "simulation": {"horizon": 1000},
            "logging": {"save_images": False, "save_states_only": True}
        }
    
    def _print_actuator_diagnostics(self):
        """Print comprehensive actuator diagnostics"""
        print(f"\nControl System Diagnostics")
        print("=" * 60)
        print(f"Explicit actuators (na): {self.model.na}")
        print(f"Control inputs (nu): {self.model.nu}")
        print(f"Joints (njnt): {self.model.njnt}")
        print(f"DOF (nv): {self.model.nv}")
        
        if self.model.nu == 0:
            print("No control inputs available!")
            return
        elif self.model.na == 0 and self.model.nu > 0:
            print("Using implicit joint controls (no explicit actuators)")
        
        print(f"\nControl system details:")
        print(f"  Control array size: {self.model.nu}")
        if hasattr(self.model, 'actuator_forcerange'):
            print(f"  Force range array: {self.model.actuator_forcerange.shape}")
            
        # Show joint names and their control indices
        print(f"\nJoint-to-control mapping:")
        for i in range(min(self.model.njnt, 10)):  # Limit to first 10 joints
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}"
            if i < self.model.nu:
                force_range = self.model.actuator_forcerange[i] if i < len(self.model.actuator_forcerange) else [0, 0]
                print(f"  [{i:2d}] {joint_name:25s} -> ctrl[{i}] Range: [{force_range[0]:6.1f}, {force_range[1]:6.1f}]")
            else:
                print(f"  [{i:2d}] {joint_name:25s} -> No control")
        
        if self.model.njnt > 10:
            print(f"  ... and {self.model.njnt - 10} more joints")
        
        print("=" * 60)
    
    def _enable_actuators(self, include_prefixes: list = ["j_", "arm_"]):
        """
        Enable control inputs for joints matching specified prefixes.
        
        Args:
            include_prefixes: List of prefixes to match for enabling joint controls
        """
        if self.model.nu == 0:
            print("No control inputs available!")
            return
        
        enabled_count = 0
        arm_controls = []
        
        print(f"\nEnabling joint controls with prefixes: {include_prefixes}")
        
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}"
            
            # Check if joint matches any of the include prefixes
            is_arm_joint = any(joint_name.startswith(prefix) for prefix in include_prefixes)
            
            if is_arm_joint and i < self.model.nu:
                # This joint has a corresponding control input
                force_range = self.model.actuator_forcerange[i] if i < len(self.model.actuator_forcerange) else [0, 0]
                
                if force_range[0] == force_range[1] == 0:
                    print(f"Joint {joint_name} has zero force range")
                else:
                    enabled_count += 1
                    arm_controls.append((i, joint_name))
                    print(f"Enabled: {joint_name} -> ctrl[{i}]")
        
        self.enabled_arm_controls = arm_controls
        
        if enabled_count == 0:
            print("WARNING: No arm controls were enabled! The robot arm may not move.")
            print("   Check that the XML file contains joints with the expected prefixes.")
        else:
            print(f"Successfully enabled {enabled_count} arm controls")
    
    def _disable_non_arm_actuators(self):
        """
        Track non-arm controls (doors, etc.) for behavior tree usage.
        All controls remain enabled for flexibility.
        """
        if self.model.nu == 0:
            return
        
        # Track which controls are NOT arm controls
        non_arm_controls = []
        
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}"
            
            # Check if it's NOT an arm joint
            is_arm_joint = any(joint_name.startswith(prefix) for prefix in ["j_", "arm_"])
            
            if not is_arm_joint and i < self.model.nu:
                non_arm_controls.append((i, joint_name))
        
        self.non_arm_controls = non_arm_controls
        
        if non_arm_controls:
            print(f"Tracked {len(non_arm_controls)} non-arm controls (doors, etc.)")
            for i, name in non_arm_controls:
                print(f"   [{i:2d}] {name}")
        
        print("All controls remain enabled for behavior tree usage")
    
    def _check_actuators_and_exit(self):
        """Print detailed control system information and exit (for --check-actuators flag)"""
        print(f"CONTROL SYSTEM CHECK MODE")
        print("=" * 70)
        
        print(f"Model Summary:")
        print(f"  Explicit actuators (na): {self.model.na}")
        print(f"  Control inputs (nu): {self.model.nu}")
        print(f"  Joints (njnt): {self.model.njnt}")
        print(f"  DOF (nv): {self.model.nv}")
        
        if self.model.nu == 0:
            print("No control inputs available!")
            sys.exit(1)
        
        print()
        print("JOINT-CONTROL MAPPING:")
        print("-" * 70)
        print(f"{'ID':<4} {'Joint Name':<25} {'Type':<12} {'Force Range':<20} {'Ctrl':<6} {'Active'}")
        print("-" * 70)
        
        arm_count = 0
        door_count = 0
        other_count = 0
        
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}"
            
            # Determine type
            joint_type = "Other"
            if joint_name.startswith(("j_", "arm_")):
                joint_type = "ARM"
                arm_count += 1
            elif "door" in joint_name.lower():
                joint_type = "DOOR"
                door_count += 1
            else:
                other_count += 1
            
            # Check if this joint has a control input
            has_control = i < self.model.nu
            ctrl_str = f"[{i}]" if has_control else "N/A"
            
            if has_control and i < len(self.model.actuator_forcerange):
                force_range = self.model.actuator_forcerange[i]
                is_active = not (force_range[0] == force_range[1] == 0)
                active_status = "YES" if is_active else "NO"
                range_str = f"[{force_range[0]:6.1f}, {force_range[1]:6.1f}]"
            else:
                active_status = "NO"
                range_str = "N/A"
            
            print(f"{i:<4} {joint_name:<25} {joint_type:<12} {range_str:<20} {ctrl_str:<6} {active_status}")
        
        print("-" * 70)
        print(f"SUMMARY:")
        print(f"  ARM joints:      {arm_count}")
        print(f"  DOOR joints:     {door_count}")
        print(f"  OTHER joints:    {other_count}")
        print(f"  TOTAL joints:    {self.model.njnt}")
        print(f"  CONTROLS:        {self.model.nu}")
        
        if arm_count == 0:
            print("\nWARNING: No ARM joints detected!")
            print("   Expected joints with prefixes: 'j_', 'arm_'")
            print("   The robot arm may not move during simulation.")
        else:
            print(f"\nFound {arm_count} ARM joints with controls - robot should move properly")
        
        if self.model.nu > 0:
            print(f"\nControl system is functional:")
            print(f"   - {self.model.nu} control inputs available")
            print(f"   - Robot can be controlled via data.ctrl array")
            print(f"   - Implicit joint controls active (even without explicit actuators)")
        
        print("\nControl system check complete. Exiting...")
        sys.exit(0)
    
    def _reset_environment(self) -> Dict[str, Any]:
        """Reset the environment to initial state."""
        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Set to home position if available
        if hasattr(self.model, 'key_qpos') and self.model.nkey > 0:
            self.data.qpos[:] = self.model.key_qpos[0]
        
        # Step once to stabilize
        mujoco.mj_step(self.model, self.data)
        
        # Reset behavior tree
        self.behavior_tree.reset()
        
        # Create initial observation
        obs = self._get_observation()
        
        print("Environment reset complete")
        return obs
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation from the environment."""
        # Robot joint positions and velocities
        robot_qpos = self.data.qpos[:7].copy()  # First 7 joints are robot
        robot_qvel = self.data.qvel[:7].copy()
        
        # Gripper state (placeholder)
        gripper_pos = np.array([0.0, 0.0])
        
        # Door positions
        fridge_door_pos = 0.0
        microwave_door_pos = 0.0
        
        # Try to get door positions from actuators
        if self.model.na > 7:
            microwave_door_pos = self.data.ctrl[7] if len(self.data.ctrl) > 7 else 0.0
        if self.model.na > 8:
            fridge_door_pos = self.data.ctrl[8] if len(self.data.ctrl) > 8 else 0.0
        
        # Object position (box)
        object_pos = np.array([1.0, 0.9, 0.55])  # Default box position in fridge
        object_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        obs = {
            "robot_joint_pos": robot_qpos,
            "robot_joint_vel": robot_qvel,
            "gripper_pos": gripper_pos,
            "fridge_door_pos": fridge_door_pos,
            "microwave_door_pos": microwave_door_pos,
            "object_pos": object_pos,
            "object_quat": object_quat,
            "timestamp": time.time()
        }
        
        return obs
    
    def _apply_action(self, action: np.ndarray) -> None:
        """Apply action to the environment with actuator-aware clipping."""
        # Ensure action is the right size for all actuators
        if len(action) > len(self.data.ctrl):
            action = action[:len(self.data.ctrl)]
        elif len(action) < len(self.data.ctrl):
            # Pad with zeros
            padded_action = np.zeros(len(self.data.ctrl))
            padded_action[:len(action)] = action
            action = padded_action
        
        # Apply action with proper force range clipping
        for i in range(len(self.data.ctrl)):
            self.data.ctrl[i] = action[i]
            
            # Clip to force range for safety (now that we have proper ranges)
            if i < len(self.model.actuator_forcerange):
                force_range = self.model.actuator_forcerange[i]
                # Only clip if we have non-zero force ranges
                if force_range[0] != 0 or force_range[1] != 0:
                    self.data.ctrl[i] = np.clip(action[i], force_range[0], force_range[1])
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Sync with viewer if available
        if self.viewer is not None:
            self.viewer.sync()
            # Add small delay for smooth visualization
            time.sleep(0.01)
    
    def collect_episode(self, episode_num: int) -> bool:
        """Collect a single trajectory episode."""
        print(f"\nStarting Episode {episode_num}")
        
        # Start logging
        trajectory_file = self.logger.start_episode()
        
        # Reset environment
        obs = self._reset_environment()
        
        # Run episode
        max_steps = self.config.get('simulation', {}).get('horizon', 1000)
        step_count = 0
        success = False
        
        for t in range(max_steps):
            # Get action from behavior tree
            action, info = self.behavior_tree.get_next_action(obs)
            
            # Apply action to environment
            self._apply_action(action)
            
            # Get new observation
            obs = self._get_observation()
            
            # Add reward and done info
            reward = self._calculate_reward(info)
            done = self.behavior_tree.is_task_done()
            success = self.behavior_tree.is_task_successful()
            
            # Enhanced info for logging
            enhanced_info = {
                **info,
                "reward": reward,
                "done": done,
                "success": success,
                "episode": episode_num,
                "total_steps": t + 1,
                "behavior_tree_progress": self.behavior_tree.get_progress()
            }
            
            # Log step
            self.logger.log_step(t, action, obs, enhanced_info)
            
            step_count += 1
            
            # Print progress every 50 steps
            if (t + 1) % 50 == 0:
                progress = self.behavior_tree.get_progress() * 100
                state = info.get('state', 'unknown')
                print(f"   Step {t+1:3d}: {state:20s} ({progress:5.1f}% complete)")
            
            # Check if episode is done
            if done:
                print(f"Episode completed in {step_count} steps")
                break
        
        # End episode logging
        metadata = {
            "scripted": True,
            "task": "fridge_to_microwave",
            "episode_number": episode_num,
            "total_steps": step_count,
            "max_steps": max_steps,
            "final_progress": self.behavior_tree.get_progress(),
            "success_flags": self.behavior_tree.success_flags.copy(),
            "final_state": self.behavior_tree.current_state.value,
            "config_file": self.config_path,
            "scene_file": self.scene_xml,
            "collection_time": datetime.now().isoformat()
        }
        
        trajectory_file = self.logger.end_episode(success=success, metadata=metadata)
        
        # Print episode summary
        status = "SUCCESS" if success else "FAILED"
        progress = self.behavior_tree.get_progress() * 100
        print(f"Episode {episode_num}: {status} ({progress:.1f}% complete)")
        print(f"Trajectory saved to: {os.path.basename(trajectory_file)}")
        
        return success
    
    def _calculate_reward(self, info: Dict[str, Any]) -> float:
        """Calculate reward based on current state and progress."""
        # Base reward on task progress
        progress = info.get('task_progress', 0.0)
        
        # Bonus for completing sub-tasks
        success_count = sum(1 for flag in self.behavior_tree.success_flags.values() if flag)
        total_tasks = len(self.behavior_tree.success_flags)
        completion_bonus = success_count / total_tasks * 0.5
        
        # Small step penalty to encourage efficiency
        step_penalty = -0.001
        
        reward = progress + completion_bonus + step_penalty
        return max(0.0, min(1.0, reward))  # Clamp to [0, 1]
    
    def collect_dataset(self, num_episodes: int = 20) -> Dict[str, Any]:
        """Collect a dataset of scripted demonstration trajectories."""
        print(f"Starting scripted data collection")
        print(f"   Episodes: {num_episodes}")
        print(f"   Task: {self.config.get('task', {}).get('name', 'fridge_to_microwave')}")
        print(f"   Output: {self.logger.session_dir}")
        print("=" * 60)
        
        start_time = time.time()
        successful_episodes = 0
        failed_episodes = 0
        
        for episode in range(1, num_episodes + 1):
            try:
                success = self.collect_episode(episode)
                if success:
                    successful_episodes += 1
                else:
                    failed_episodes += 1
                    
            except KeyboardInterrupt:
                print(f"\nCollection interrupted by user at episode {episode}")
                break
            except Exception as e:
                print(f"Error in episode {episode}: {e}")
                failed_episodes += 1
                continue
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate summary statistics
        total_episodes = successful_episodes + failed_episodes
        success_rate = successful_episodes / total_episodes if total_episodes > 0 else 0.0
        
        summary = {
            "total_episodes": total_episodes,
            "successful_episodes": successful_episodes,
            "failed_episodes": failed_episodes,
            "success_rate": success_rate,
            "duration_seconds": duration,
            "session_dir": self.logger.session_dir
        }
        
        # Print final summary
        print("\n" + "=" * 60)
        print(f"Collection Summary")
        print(f"   Total episodes: {total_episodes}")
        print(f"   Successful: {successful_episodes}")
        print(f"   Failed: {failed_episodes}")
        print(f"   Success rate: {success_rate*100:.1f}%")
        print(f"   Duration: {duration:.1f} seconds")
        print(f"   Dataset saved to: {self.logger.session_dir}")
        
        return summary
    
    def close(self):
        """Clean up resources, especially the viewer."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
            print("Viewer closed")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Collect scripted kitchen manipulation demonstrations')
    parser.add_argument('--episodes', type=int, default=20,
                        help='Number of episodes to collect (default: 20)')
    parser.add_argument('--config', type=str, default='config/scene.yaml',
                        help='Path to scene configuration file')
    parser.add_argument('--scene', type=str, default='custom_kitchen.xml',
                        help='Path to MuJoCo scene XML file')
    parser.add_argument('--visual', action='store_true', default=False,
                        help='Run with visual rendering (opens MuJoCo viewer window)')
    parser.add_argument('--headless', action='store_true', default=False,
                        help='Run in headless mode (no visual rendering)')
    parser.add_argument('--check-actuators', action='store_true', default=False,
                        help='Print detailed actuator information and exit')
    
    args = parser.parse_args()
    
    # Determine headless mode: default to headless unless --visual is specified
    headless = not args.visual if args.visual else not args.headless if args.headless else True
    
    print("Scripted Demo Collector for Kitchen Manipulation")
    print("=" * 60)
    
    collector = None
    try:
        # Initialize collector
        collector = ScriptedDemoCollector(
            config_path=args.config,
            scene_xml=args.scene,
            headless=headless
        )
        
        # Handle --check-actuators flag
        if getattr(args, 'check_actuators', False):
            collector._check_actuators_and_exit()
        
        # Show visual mode status
        mode = "HEADLESS" if headless else "VISUAL"
        print(f"Running in {mode} mode")
        
        # Collect dataset
        summary = collector.collect_dataset(num_episodes=args.episodes)
        
        # Print final result
        print(f"\nData collection completed!")
        print(f"Trajectory saved to {summary['session_dir']}")
        
        return summary
    
    except KeyboardInterrupt:
        print(f"\nCollection interrupted by user")
        return None
    except Exception as e:
        print(f"Error during collection: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Clean up viewer
        if collector is not None:
            collector.close()


if __name__ == "__main__":
    import sys
    
    # Check if command-line arguments are provided
    if len(sys.argv) > 1:
        # Use main() function for command-line usage
        main()
    else:
        # Minimal working main that runs the scripted plan once
        print("Running minimal working example...")
        
        try:
            # Initialize collector with default settings
            collector = ScriptedDemoCollector()
            
            # Run single episode
            print("\nRunning single scripted episode...")
            success = collector.collect_episode(1)
            
            # Get session info
            session_info = collector.logger.get_session_info()
            
            if success:
                print(f"\nSUCCESS: Trajectory saved to {session_info['session_dir']}/traj_0001.jsonl")
            else:
                print(f"\nFAILED: Trajectory saved to {session_info['session_dir']}/traj_0001.jsonl")
                
            print(f"Full dataset location: {session_info['session_dir']}")
            
        except Exception as e:
            print(f"SError: {e}")
            import traceback
            traceback.print_exc()