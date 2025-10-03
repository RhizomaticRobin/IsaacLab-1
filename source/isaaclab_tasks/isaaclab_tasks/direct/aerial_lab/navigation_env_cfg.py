# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration for the aerial navigation task with vision and domain randomization.

This environment is a conversion from the aerial_gym_simulator navigation task.
It features:
- Vision-based navigation using depth cameras
- Extensive domain randomization (robot dynamics, sensors, obstacles)
- Curriculum learning for obstacle density
- Motor model simulation
"""

from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import os

##
# Scene definition
##


@configclass
class AerialNavigationSceneCfg(InteractiveSceneCfg):
    """Configuration for the aerial navigation scene."""

    # Ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Quadrotor robot (LMF2)
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=f"{os.path.dirname(__file__)}/lmf2/model.urdf",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=100.0,
                max_angular_velocity=100.0,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
        actuators={},  # No joint actuators - we use direct force/torque control
    )

    # Depth camera sensor
    depth_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/depth_camera",
        update_period=0.1,  # 10 Hz
        height=135,
        width=240,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.2, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.10, 0.0, 0.03),
            rot=(0.7071068, 0.0, 0.0, 0.7071068),  # -90 deg around X to face forward
            convention="world",
        ),
    )

    # Lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(color=(0.9, 0.9, 0.9), intensity=600.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.738, 0.477, 0.477, 0.0)),
    )


##
# MDP settings
##


@configclass
class EventCfg:
    """Configuration for domain randomization events."""

    # Robot state randomization at reset
    reset_robot_state = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-1.0, 1.0),  # Will be scaled by environment bounds
                "y": (-1.0, 1.0),
                "z": (0.5, 2.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-math.pi / 6, math.pi / 6),
            },
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        },
    )

    # Random pushes/disturbances during flight
    apply_disturbance = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(0.1, 0.2),  # Apply every 0.1-0.2 seconds with probability
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "force_range": (-4.75, 4.75),
            "torque_range": (-0.03, 0.03),
            "probability": 0.05,
        },
    )

    # Randomize rigid body mass at startup
    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-0.05, 0.05),
            "operation": "scale",
        },
    )

    # Randomize damping coefficients at startup
    randomize_joint_parameters = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.8, 1.2),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )


##
# Environment configuration
##


@configclass
class AerialNavigationEnvCfg(DirectRLEnvCfg):
    """Configuration for the aerial navigation environment."""

    # Environment settings
    episode_length_s = 10.0  # 100 steps * 0.1s
    decimation = 2
    action_space = 4  # [vx, vy, vz, yaw_rate] velocity commands
    observation_space = 13 + 4 + 64  # state (13) + action (4) + VAE latents (64)
    state_space = 0  # No privileged observations

    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 100.0,  # 100 Hz physics
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Scene settings
    scene: AerialNavigationSceneCfg = AerialNavigationSceneCfg(
        num_envs=1024, env_spacing=5.0, replicate_physics=True
    )

    # MDP settings
    events: EventCfg = EventCfg()

    # Task-specific settings
    # Environment bounds randomization
    env_lower_bound_min = [-2.0, -4.0, -3.0]
    env_lower_bound_max = [-1.0, -2.5, -2.0]
    env_upper_bound_min = [9.0, 2.5, 2.0]
    env_upper_bound_max = [10.0, 4.0, 3.0]

    # Target position (ratio within environment bounds)
    target_min_ratio = [0.90, 0.1, 0.1]
    target_max_ratio = [0.94, 0.90, 0.90]

    # Motor model parameters (will be randomized per environment)
    motor_thrust_constant_min = 0.00000926312
    motor_thrust_constant_max = 0.00001826312
    motor_time_constant_increasing_min = 0.05
    motor_time_constant_increasing_max = 0.08
    motor_time_constant_decreasing_min = 0.005
    motor_time_constant_decreasing_max = 0.005

    # Control allocation matrix (standard X-configuration quadrotor)
    # [fx, fy, fz, tx, ty, tz] = allocation_matrix @ [m1, m2, m3, m4]
    allocation_matrix = [
        [0.0, 0.0, 0.0, 0.0],  # fx
        [0.0, 0.0, 0.0, 0.0],  # fy
        [1.0, 1.0, 1.0, 1.0],  # fz (thrust)
        [-0.13, -0.13, 0.13, 0.13],  # roll torque
        [-0.13, 0.13, 0.13, -0.13],  # pitch torque
        [-0.07, 0.07, -0.07, 0.07],  # yaw torque
    ]
    motor_directions = [1, -1, 1, -1]

    # Curriculum learning settings
    curriculum_min_level = 15  # Minimum number of obstacles
    curriculum_max_level = 50  # Maximum number of obstacles
    curriculum_check_after_instances = 2048
    curriculum_increase_step = 2
    curriculum_decrease_step = 1
    curriculum_success_rate_for_increase = 0.7
    curriculum_success_rate_for_decrease = 0.6

    # Reward weights
    pos_reward_magnitude = 5.0
    pos_reward_exponent = 1.0 / 3.5
    very_close_reward_magnitude = 5.0
    very_close_reward_exponent = 2.0
    getting_closer_multiplier = 10.0
    collision_penalty = -100.0

    # Action penalties
    x_action_diff_penalty_magnitude = 0.8
    x_action_diff_penalty_exponent = 3.333
    z_action_diff_penalty_magnitude = 0.8
    z_action_diff_penalty_exponent = 5.0
    yawrate_action_diff_penalty_magnitude = 0.8
    yawrate_action_diff_penalty_exponent = 3.33

    x_absolute_action_penalty_magnitude = 0.1
    x_absolute_action_penalty_exponent = 0.3
    z_absolute_action_penalty_magnitude = 1.5
    z_absolute_action_penalty_exponent = 1.0
    yawrate_absolute_action_penalty_magnitude = 1.5
    yawrate_absolute_action_penalty_exponent = 2.0

    # Image-based obstacle avoidance reward
    min_pixel_distance_penalty_magnitude = 4.0
    min_pixel_distance_penalty_exponent = 1.0

    # VAE settings (if using pre-trained VAE for depth encoding)
    use_vae = False  # Set to True if you have a pre-trained VAE
    vae_latent_dims = 64
    vae_model_path = None  # Path to pre-trained VAE weights

    # Success threshold
    success_distance_threshold = 1.0

    def __post_init__(self):
        """Post initialization."""
        # Set viewer settings
        self.viewer.eye = (10.0, 10.0, 5.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)


# Import MDP functions (will be created)
# Note: These would need to be implemented as custom functions
# For now, we'll use placeholder imports
try:
    import isaaclab.envs.mdp as mdp
except ImportError:
    # Placeholder for custom MDP functions
    class mdp:
        reset_root_state_uniform = None
        apply_external_force_torque = None
        randomize_rigid_body_mass = None
        randomize_rigid_body_material = None
