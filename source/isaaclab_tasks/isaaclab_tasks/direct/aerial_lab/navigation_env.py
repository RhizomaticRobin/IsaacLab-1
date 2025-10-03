# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Aerial navigation environment with vision and domain randomization.

Converted from aerial_gym_simulator navigation task.
Features motor dynamics simulation, depth-based obstacle avoidance, and curriculum learning.
"""

import gymnasium as gym
import math
import torch
from collections.abc import Sequence
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import Camera
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import (
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
    subtract_frame_transforms,
)

from .navigation_env_cfg import AerialNavigationEnvCfg


class AerialNavigationEnv(DirectRLEnv):
    """Aerial navigation environment with vision-based obstacle avoidance.

    This environment implements:
    - Vision-based navigation using depth cameras (with optional VAE encoding)
    - Motor dynamics simulation with randomization
    - Extensive domain randomization (mass, inertia, disturbances, camera placement)
    - Curriculum learning for obstacle density
    - Complex reward function balancing goal-reaching and obstacle avoidance
    """

    cfg: AerialNavigationEnvCfg

    def __init__(self, cfg: AerialNavigationEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the environment."""
        super().__init__(cfg, render_mode, **kwargs)

        # Robot mass for thrust calculation
        self._robot_mass = None
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()

        # Environment bounds (randomized per environment)
        self._env_lower_bound = torch.zeros((self.num_envs, 3), device=self.device)
        self._env_upper_bound = torch.zeros((self.num_envs, 3), device=self.device)

        # Target positions
        self._target_position = torch.zeros((self.num_envs, 3), device=self.device)

        # Motor model state
        self._motor_rpm = torch.zeros((self.num_envs, 4), device=self.device)
        self._motor_thrust = torch.zeros((self.num_envs, 4), device=self.device)
        self._motor_thrust_constants = torch.zeros((self.num_envs, 4), device=self.device)
        self._motor_time_constants_inc = torch.zeros((self.num_envs, 4), device=self.device)
        self._motor_time_constants_dec = torch.zeros((self.num_envs, 4), device=self.device)

        # Control allocation matrix [fx, fy, fz, tx, ty, tz] = A @ [m1, m2, m3, m4]
        self._allocation_matrix = torch.tensor(
            self.cfg.allocation_matrix, dtype=torch.float32, device=self.device
        )

        # Forces and torques to apply
        self._forces = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self._torques = torch.zeros((self.num_envs, 1, 3), device=self.device)

        # Previous observations for reward computation
        self._prev_pos_error = torch.zeros((self.num_envs, 3), device=self.device)
        self._prev_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        # Curriculum learning
        self._curriculum_level = self.cfg.curriculum_min_level
        self._success_count = 0
        self._crash_count = 0
        self._timeout_count = 0

        # Image observations (if using depth camera)
        self._depth_latents = torch.zeros((self.num_envs, self.cfg.vae_latent_dims), device=self.device)

        # VAE model (if enabled)
        self._vae_model = None
        if self.cfg.use_vae and self.cfg.vae_model_path is not None:
            self._load_vae_model()

        # Initialize motor model parameters
        self._initialize_motor_parameters()

    def _setup_scene(self):
        """Setup the scene with robot, sensors, and lights."""
        # Add robot
        self._robot = Articulation(self.cfg.scene.robot)
        self.scene.articulations["robot"] = self._robot

        # Add depth camera
        self._depth_camera = Camera(self.cfg.scene.depth_camera)
        self.scene.sensors["depth_camera"] = self._depth_camera

        # Add lights
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        # Add terrain/ground
        self.cfg.scene.terrain.prim_path = "/World/ground"
        self.cfg.scene.terrain.func("/World/ground", self.cfg.scene.terrain)

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions before physics step.

        Actions are velocity commands [vx, vy, vz, yaw_rate] that get transformed
        into motor commands through the motor model.
        """
        # Transform actions to velocity commands
        # Actions are in [-1, 1], scale to actual velocity/yaw rate
        max_speed = 2.0  # m/s
        max_yaw_rate = math.pi / 3  # rad/s
        max_inclination = math.pi / 4  # rad

        # Transform actions: action[0] is forward speed + pitch command
        # action[1] is lateral tilt, action[2] is yaw rate
        velocity_commands = torch.zeros((self.num_envs, 4), device=self.device)

        # Forward velocity with inclination
        velocity_commands[:, 0] = (
            (actions[:, 0] + 1.0)
            * torch.cos(max_inclination * actions[:, 1])
            * max_speed
            / 2.0
        )
        velocity_commands[:, 1] = 0.0  # No lateral velocity
        velocity_commands[:, 2] = (
            (actions[:, 0] + 1.0)
            * torch.sin(max_inclination * actions[:, 1])
            * max_speed
            / 2.0
        )
        velocity_commands[:, 3] = actions[:, 2] * max_yaw_rate

        # Convert velocity commands to desired thrust using simplified controller
        # This is a placeholder - in full implementation, this would use the Lee controller
        thrust_command = self._robot_mass * self._gravity_magnitude  # Hover thrust
        roll_command = velocity_commands[:, 1] * 0.1  # Simplified
        pitch_command = -velocity_commands[:, 0] * 0.1  # Simplified (negative for forward)
        yaw_rate_command = velocity_commands[:, 3]

        # Convert to motor commands using allocation matrix inverse
        desired_wrench = torch.stack(
            [
                torch.zeros_like(thrust_command),  # fx
                torch.zeros_like(thrust_command),  # fy
                thrust_command.expand(self.num_envs),  # fz
                roll_command,  # tx
                pitch_command,  # ty
                yaw_rate_command * 0.01,  # tz (scaled)
            ],
            dim=1,
        )

        # Pseudo-inverse to get motor thrusts
        # Note: This is simplified. Full implementation would use proper control allocation
        motor_thrusts = torch.linalg.lstsq(
            self._allocation_matrix.T, desired_wrench.T
        ).solution.T

        # Clamp motor thrusts to valid range
        motor_thrusts = torch.clamp(motor_thrusts, 0.1, 10.0)

        # Update motor dynamics (first-order model)
        self._update_motor_dynamics(motor_thrusts)

        # Store previous action for reward
        self._prev_actions = actions.clone()

    def _apply_action(self):
        """Apply forces and torques computed from motor model.

        This is called every simulation step (potentially multiple times per env step).
        """
        # Compute wrench from motor thrusts
        wrench = torch.matmul(self._motor_thrust.unsqueeze(1), self._allocation_matrix.T).squeeze(1)

        # Extract forces and torques
        self._forces[:, 0, :] = wrench[:, 0:3]
        self._torques[:, 0, :] = wrench[:, 3:6]

        # Apply to robot
        self._robot.set_external_force_and_torque(
            self._forces, self._torques, body_ids=[0]  # Apply to base link
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Compute observations."""
        # Get robot state
        robot_pos = self._robot.data.root_pos_w
        robot_quat = self._robot.data.root_quat_w
        robot_lin_vel_b = self._robot.data.root_lin_vel_b
        robot_ang_vel_b = self._robot.data.root_ang_vel_b

        # Compute position error in vehicle frame
        pos_error = self._target_position - robot_pos
        pos_error_vehicle = quat_rotate_inverse(robot_quat, pos_error)

        # Distance and direction to target
        dist_to_target = torch.norm(pos_error, dim=1, keepdim=True)
        unit_vec_to_target = pos_error_vehicle / (dist_to_target + 1e-6)

        # Add perturbation for robustness
        perturbed_unit_vec = unit_vec_to_target + 0.1 * (torch.rand_like(unit_vec_to_target) - 0.5)
        perturbed_unit_vec = perturbed_unit_vec / (torch.norm(perturbed_unit_vec, dim=1, keepdim=True) + 1e-6)

        # Euler angles
        euler_angles = self._quat_to_euler(robot_quat)

        # Construct state observation [13 dims]
        state_obs = torch.cat(
            [
                perturbed_unit_vec,  # 3: direction to target
                dist_to_target,  # 1: distance to target
                euler_angles[:, 0:2],  # 2: roll, pitch
                robot_lin_vel_b,  # 3: body frame linear velocity
                robot_ang_vel_b,  # 3: body frame angular velocity
                self._prev_actions,  # 4: previous actions
            ],
            dim=1,
        )

        # Process depth image if available
        if self._depth_camera.is_initialized:
            self._process_depth_image()

        # Concatenate state and vision observations
        full_obs = torch.cat([state_obs, self._depth_latents], dim=1)

        return {"policy": full_obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards."""
        # Get current state
        robot_pos = self._robot.data.root_pos_w
        robot_quat = self._robot.data.root_quat_w

        # Position error in vehicle frame
        pos_error = self._target_position - robot_pos
        pos_error_vehicle = quat_rotate_inverse(robot_quat, pos_error)
        dist = torch.norm(pos_error_vehicle, dim=1)

        # Previous distance
        prev_dist = torch.norm(self._prev_pos_error, dim=1)

        # Position reward (exponential)
        pos_reward = self.cfg.pos_reward_magnitude * torch.exp(
            -(dist**2) * self.cfg.pos_reward_exponent
        )

        # Very close to goal reward
        very_close_reward = self.cfg.very_close_reward_magnitude * torch.exp(
            -(dist**2) * self.cfg.very_close_reward_exponent
        )

        # Getting closer reward
        getting_closer = prev_dist - dist
        getting_closer_reward = torch.where(
            getting_closer > 0,
            self.cfg.getting_closer_multiplier * getting_closer,
            2.0 * self.cfg.getting_closer_multiplier * getting_closer,
        )

        # Distance-based reward
        distance_reward = (20.0 - dist) / 20.0

        # Action smoothness penalty
        action_diff = self.actions - self._prev_actions
        action_diff_penalty = self._exponential_penalty(
            self.cfg.x_action_diff_penalty_magnitude,
            self.cfg.x_action_diff_penalty_exponent,
            action_diff[:, 0],
        )

        # Curriculum progress (0 to 1)
        curriculum_progress = (self._curriculum_level - self.cfg.curriculum_min_level) / (
            self.cfg.curriculum_max_level - self.cfg.curriculum_min_level
        )

        # Total reward
        reward = (
            (1.0 + 2.0 * curriculum_progress)
            * (pos_reward + very_close_reward + getting_closer_reward + distance_reward)
            + action_diff_penalty
        )

        # Image-based obstacle avoidance penalty (if using depth camera)
        if self._depth_camera.is_initialized:
            # Get minimum depth in image
            depth_image = self._depth_camera.data.output["distance_to_image_plane"]
            min_depth = torch.amin(depth_image, dim=(1, 2, 3))
            obstacle_penalty = -torch.exp(-(min_depth**2) * self.cfg.min_pixel_distance_penalty_exponent)
            reward += obstacle_penalty * self.cfg.min_pixel_distance_penalty_magnitude

        # Update previous position error
        self._prev_pos_error = pos_error_vehicle.clone()

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute termination and truncation flags."""
        # Timeout
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Collision detection (simplified - check if robot is too low or contacts ground)
        died = self._robot.data.root_pos_w[:, 2] < 0.1

        # Check if out of bounds
        out_of_bounds = torch.any(
            torch.logical_or(
                self._robot.data.root_pos_w < self._env_lower_bound,
                self._robot.data.root_pos_w > self._env_upper_bound,
            ),
            dim=1,
        )
        died = torch.logical_or(died, out_of_bounds)

        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset specified environments."""
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        # Reset robot state with randomization
        self._reset_robot_state(env_ids)

        # Reset motor model
        self._reset_motor_model(env_ids)

        # Reset environment bounds
        self._reset_environment_bounds(env_ids)

        # Reset target position
        self._reset_target_position(env_ids)

        # Reset curriculum learning counters (update based on success rate)
        # This is done globally, not per environment
        if len(env_ids) > 100:  # Only update curriculum on batch resets
            self._update_curriculum()

        # Reset previous observations
        self._prev_pos_error[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0

        # Reset episode tracking
        super()._reset_idx(env_ids)

    ##
    # Helper methods
    ##

    def _initialize_motor_parameters(self):
        """Initialize motor model parameters with randomization."""
        # Randomize motor thrust constants
        self._motor_thrust_constants = torch.rand((self.num_envs, 4), device=self.device) * (
            self.cfg.motor_thrust_constant_max - self.cfg.motor_thrust_constant_min
        ) + self.cfg.motor_thrust_constant_min

        # Randomize time constants
        self._motor_time_constants_inc = torch.rand((self.num_envs, 4), device=self.device) * (
            self.cfg.motor_time_constant_increasing_max - self.cfg.motor_time_constant_increasing_min
        ) + self.cfg.motor_time_constant_increasing_min

        self._motor_time_constants_dec = torch.rand((self.num_envs, 4), device=self.device) * (
            self.cfg.motor_time_constant_decreasing_max - self.cfg.motor_time_constant_decreasing_min
        ) + self.cfg.motor_time_constant_decreasing_min

        # Initialize motor states
        self._motor_rpm.zero_()
        self._motor_thrust.zero_()

        # Get robot mass
        if self._robot_mass is None:
            self._robot_mass = self._robot.root_physx_view.get_masses().sum(dim=1)

    def _update_motor_dynamics(self, desired_thrust: torch.Tensor):
        """Update motor dynamics using first-order model.

        Args:
            desired_thrust: Desired thrust for each motor [N]
        """
        # First-order motor dynamics: tau * d(thrust)/dt + thrust = desired_thrust
        # Discrete approximation: thrust[k+1] = thrust[k] + dt/tau * (desired - thrust[k])

        dt = self.cfg.sim.dt * self.cfg.decimation

        # Choose time constant based on increasing or decreasing
        time_constant = torch.where(
            desired_thrust > self._motor_thrust,
            self._motor_time_constants_inc,
            self._motor_time_constants_dec,
        )

        # Update thrust with first-order dynamics
        alpha = dt / (time_constant + dt)
        self._motor_thrust = self._motor_thrust + alpha * (desired_thrust - self._motor_thrust)

        # Clamp to valid range
        self._motor_thrust = torch.clamp(self._motor_thrust, 0.1, 10.0)

    def _reset_robot_state(self, env_ids: torch.Tensor):
        """Reset robot state with randomization."""
        # Sample random position within environment bounds
        pos_ratio = torch.rand((len(env_ids), 3), device=self.device)
        pos_ratio[:, 0] = pos_ratio[:, 0] * 0.1 + 0.1  # Start near beginning
        pos_ratio[:, 1] = pos_ratio[:, 1] * 0.7 + 0.15  # Middle range
        pos_ratio[:, 2] = pos_ratio[:, 2] * 0.7 + 0.15  # Middle range

        # Interpolate within bounds
        pos = (
            self._env_lower_bound[env_ids]
            + pos_ratio * (self._env_upper_bound[env_ids] - self._env_lower_bound[env_ids])
        )

        # Random orientation (yaw only)
        yaw = torch.rand((len(env_ids),), device=self.device) * (math.pi / 3) - (math.pi / 6)
        quat = quat_from_euler_xyz(
            torch.zeros_like(yaw), torch.zeros_like(yaw), yaw
        )

        # Random velocities
        lin_vel = (torch.rand((len(env_ids), 3), device=self.device) - 0.5) * 0.4
        ang_vel = (torch.rand((len(env_ids), 3), device=self.device) - 0.5) * 0.4

        # Set state
        self._robot.write_root_pose_to_sim(
            torch.cat([pos, quat], dim=1), env_ids
        )
        self._robot.write_root_velocity_to_sim(
            torch.cat([lin_vel, ang_vel], dim=1), env_ids
        )

    def _reset_motor_model(self, env_ids: torch.Tensor):
        """Reset motor model with randomization."""
        # Re-randomize motor parameters
        self._motor_thrust_constants[env_ids] = torch.rand((len(env_ids), 4), device=self.device) * (
            self.cfg.motor_thrust_constant_max - self.cfg.motor_thrust_constant_min
        ) + self.cfg.motor_thrust_constant_min

        self._motor_time_constants_inc[env_ids] = torch.rand((len(env_ids), 4), device=self.device) * (
            self.cfg.motor_time_constant_increasing_max - self.cfg.motor_time_constant_increasing_min
        ) + self.cfg.motor_time_constant_increasing_min

        self._motor_time_constants_dec[env_ids] = torch.rand((len(env_ids), 4), device=self.device) * (
            self.cfg.motor_time_constant_decreasing_max - self.cfg.motor_time_constant_decreasing_min
        ) + self.cfg.motor_time_constant_decreasing_min

        # Reset motor states
        self._motor_rpm[env_ids] = 0.0
        self._motor_thrust[env_ids] = 0.0

    def _reset_environment_bounds(self, env_ids: torch.Tensor):
        """Randomize environment bounds."""
        # Sample random bounds
        self._env_lower_bound[env_ids] = torch.rand((len(env_ids), 3), device=self.device) * torch.tensor(
            [
                self.cfg.env_lower_bound_max[i] - self.cfg.env_lower_bound_min[i]
                for i in range(3)
            ],
            device=self.device,
        ) + torch.tensor(self.cfg.env_lower_bound_min, device=self.device)

        self._env_upper_bound[env_ids] = torch.rand((len(env_ids), 3), device=self.device) * torch.tensor(
            [
                self.cfg.env_upper_bound_max[i] - self.cfg.env_upper_bound_min[i]
                for i in range(3)
            ],
            device=self.device,
        ) + torch.tensor(self.cfg.env_upper_bound_min, device=self.device)

    def _reset_target_position(self, env_ids: torch.Tensor):
        """Reset target position."""
        # Sample target position as ratio of environment bounds
        target_ratio = torch.rand((len(env_ids), 3), device=self.device) * torch.tensor(
            [
                self.cfg.target_max_ratio[i] - self.cfg.target_min_ratio[i]
                for i in range(3)
            ],
            device=self.device,
        ) + torch.tensor(self.cfg.target_min_ratio, device=self.device)

        # Interpolate within bounds
        self._target_position[env_ids] = (
            self._env_lower_bound[env_ids]
            + target_ratio * (self._env_upper_bound[env_ids] - self._env_lower_bound[env_ids])
        )

    def _update_curriculum(self):
        """Update curriculum level based on success rate."""
        total_instances = self._success_count + self._crash_count + self._timeout_count

        if total_instances >= self.cfg.curriculum_check_after_instances:
            success_rate = self._success_count / total_instances

            if success_rate > self.cfg.curriculum_success_rate_for_increase:
                self._curriculum_level = min(
                    self._curriculum_level + self.cfg.curriculum_increase_step,
                    self.cfg.curriculum_max_level,
                )
            elif success_rate < self.cfg.curriculum_success_rate_for_decrease:
                self._curriculum_level = max(
                    self._curriculum_level - self.cfg.curriculum_decrease_step,
                    self.cfg.curriculum_min_level,
                )

            # Reset counters
            self._success_count = 0
            self._crash_count = 0
            self._timeout_count = 0

            print(f"Curriculum level: {self._curriculum_level}, Success rate: {success_rate:.2f}")

    def _process_depth_image(self):
        """Process depth camera image, optionally encoding with VAE."""
        if not self._depth_camera.is_initialized:
            return

        # Get depth image
        depth_image = self._depth_camera.data.output["distance_to_image_plane"]

        if self._vae_model is not None:
            # Encode with VAE
            with torch.no_grad():
                self._depth_latents = self._vae_model.encode(depth_image)
        else:
            # Use raw depth statistics as features (placeholder)
            # In full implementation, you'd want to use a proper VAE or CNN encoder
            depth_flat = depth_image.view(self.num_envs, -1)
            self._depth_latents[:, 0] = torch.mean(depth_flat, dim=1)
            self._depth_latents[:, 1] = torch.std(depth_flat, dim=1)
            self._depth_latents[:, 2] = torch.amin(depth_flat, dim=1)
            # Rest stays zero

    def _load_vae_model(self):
        """Load pre-trained VAE model for depth encoding."""
        # Placeholder - would need actual VAE implementation
        print(f"VAE model loading not implemented. Path: {self.cfg.vae_model_path}")
        self._vae_model = None

    def _quat_to_euler(self, quat: torch.Tensor) -> torch.Tensor:
        """Convert quaternion to Euler angles (roll, pitch, yaw)."""
        # Implementation from aerial_gym math utils
        # quat = [w, x, y, z]
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        pitch = torch.where(
            torch.abs(sinp) >= 1,
            torch.sign(sinp) * math.pi / 2,
            torch.asin(sinp),
        )

        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        return torch.stack([roll, pitch, yaw], dim=1)

    def _exponential_penalty(
        self, magnitude: float, exponent: float, value: torch.Tensor
    ) -> torch.Tensor:
        """Compute exponential penalty."""
        return magnitude * (torch.exp(-(value**2) * exponent) - 1.0)
