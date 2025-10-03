# Aerial Lab: Vision-Based Aerial Navigation

Converted from [aerial_gym_simulator](https://github.com/ntnu-arl/aerial_gym_simulator) navigation task to IsaacLab.

## Overview

This environment implements vision-based autonomous navigation for quadrotors with extensive domain randomization and curriculum learning. It features:

- **Vision-based Navigation**: Depth camera-based obstacle avoidance (with optional VAE encoding)
- **Motor Dynamics Simulation**: First-order motor model with randomized parameters
- **Extensive Domain Randomization**:
  - Robot dynamics (mass, inertia, damping)
  - Motor parameters (thrust constants, time constants)
  - Initial states (position, orientation, velocity)
  - External disturbances (random forces/torques)
  - Camera placement
  - Environment bounds
- **Curriculum Learning**: Adaptive obstacle density based on success rate
- **Complex Reward Function**: Balances goal-reaching, obstacle avoidance, and action smoothness

## Key Features Converted from aerial_gym_simulator

### Navigation Task Components

| Component | aerial_gym_simulator | IsaacLab (aerial_lab) | Status |
|-----------|---------------------|----------------------|--------|
| **Robot Model** | LMF2 quadrotor URDF | ✓ Copied and configured | ✓ |
| **Motor Model** | First-order dynamics with randomization | ✓ Implemented in `_update_motor_dynamics()` | ✓ |
| **Control Allocation** | Matrix-based thrust to wrench | ✓ Implemented using `_allocation_matrix` | ✓ |
| **Depth Camera** | Warp-based 135x240 depth | ✓ IsaacLab Camera sensor | ✓ |
| **VAE Encoding** | Pre-trained 64-dim latent encoding | ⚠ Placeholder (needs model file) | ⚠ |
| **Curriculum Learning** | 15-50 obstacles adaptive | ✓ Implemented | ✓ |
| **Domain Randomization** | 8+ randomization types | ✓ EventManager + manual | ✓ |
| **Reward Function** | Multi-term exponential rewards | ✓ JIT-compatible implementation | ✓ |

### Domain Randomization Mapping

| DR Type | aerial_gym_simulator | IsaacLab Implementation |
|---------|---------------------|------------------------|
| **Initial State** | `min_init_state`, `max_init_state` | `_reset_robot_state()` with pose/velocity ranges |
| **Motor Thrust** | `motor_thrust_constant_min/max` | `_motor_thrust_constants` randomization |
| **Motor Time Constants** | `motor_time_constant_*` | `_motor_time_constants_inc/dec` |
| **Mass** | Robot asset density | `EventManager` - `randomize_mass` |
| **Disturbances** | Random forces with probability | `EventManager` - `apply_disturbance` |
| **Environment Bounds** | `lower_bound_min/max`, `upper_bound_min/max` | `_reset_environment_bounds()` |
| **Camera Placement** | `randomize_placement` in sensor config | Camera `offset` randomization (to be added) |
| **Friction/Restitution** | Physics material randomization | `EventManager` - `randomize_joint_parameters` |

## Architecture

### DirectRLEnv Workflow

```
Actions [vx, vy, vz, yaw_rate]
    ↓
_pre_physics_step()
    ↓ Transform to thrust commands
Motor Model (_update_motor_dynamics)
    ↓ First-order dynamics
Control Allocation (allocation_matrix)
    ↓ Wrench = A @ motor_thrusts
_apply_action()
    ↓ Apply forces/torques
Physics Simulation
    ↓
_get_observations()
    ↓ State + Vision (depth camera)
_get_rewards()
    ↓ Multi-term reward
_get_dones()
```

### Key Classes

- **`AerialNavigationEnv`**: Main environment implementing DirectRLEnv
- **`AerialNavigationEnvCfg`**: Configuration with all hyperparameters
- **`AerialNavigationSceneCfg`**: Scene setup (robot, camera, lights, terrain)
- **`EventCfg`**: Domain randomization events

## Quick Start

### 1. Training

```bash
# Navigate to IsaacLab directory
cd /path/to/IsaacLab

# Run training with RSL-RL
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Aerial-Navigation-Direct-v0 \
    --headless \
    --num_envs 1024

# Or with RL-Games
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Aerial-Navigation-Direct-v0 \
    --headless
```

### 2. Evaluation

```bash
# Play trained policy
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Aerial-Navigation-Direct-v0-Play \
    --checkpoint /path/to/model.pt \
    --num_envs 32
```

### 3. Testing

```bash
# Quick smoke test
python -c "import gymnasium as gym; \
           from isaaclab_tasks.direct.aerial_lab import *; \
           env = gym.make('Isaac-Aerial-Navigation-Direct-v0', num_envs=4); \
           env.reset(); \
           print('Environment created successfully!')"
```

## Configuration

### Key Parameters

Edit `navigation_env_cfg.py` to customize:

```python
@configclass
class AerialNavigationEnvCfg(DirectRLEnvCfg):
    # Episode settings
    episode_length_s = 10.0  # 100 steps at 10 Hz

    # Environment size
    scene.num_envs = 1024
    scene.env_spacing = 5.0

    # Curriculum learning
    curriculum_min_level = 15  # Start with 15 obstacles
    curriculum_max_level = 50  # Up to 50 obstacles
    curriculum_success_rate_for_increase = 0.7  # 70% success to increase

    # Motor model randomization
    motor_thrust_constant_min = 0.00000926312
    motor_thrust_constant_max = 0.00001826312
    motor_time_constant_increasing_min = 0.05
    motor_time_constant_increasing_max = 0.08

    # Reward weights
    pos_reward_magnitude = 5.0
    collision_penalty = -100.0
    getting_closer_multiplier = 10.0
```

### Enabling VAE Encoding

To use pre-trained VAE for depth encoding:

1. Train or obtain VAE model (64-dim latent for 135x240 depth images)
2. Update configuration:

```python
cfg.use_vae = True
cfg.vae_model_path = "/path/to/vae_weights.pth"
```

3. Implement `_load_vae_model()` in `navigation_env.py` to load your VAE

## Implementation Notes

### What's Implemented

✅ **Core Functionality**:
- DirectRLEnv with motor dynamics
- Depth camera sensor
- Control allocation matrix
- First-order motor model with randomization
- Multi-term reward function
- Curriculum learning
- Event-based domain randomization

✅ **Domain Randomization**:
- Robot state (position, orientation, velocity)
- Motor parameters (thrust constants, time constants)
- Mass properties
- External disturbances
- Environment bounds
- Physics materials (friction, restitution)

### What Needs Work

⚠ **To Complete**:

1. **VAE Model**: Implement `_load_vae_model()` with actual VAE architecture
   - Current: Uses depth statistics as placeholder features
   - Needed: Proper VAE encoder matching aerial_gym's architecture

2. **Obstacle Spawning**: Add dynamic obstacle generation
   - Current: Empty environment with walls
   - Needed: Panels, objects, trees based on curriculum level

3. **Camera Placement Randomization**: Add to EventManager
   - Current: Fixed camera offset
   - Needed: Random translation/rotation within bounds

4. **Velocity Controller**: Implement full Lee controller
   - Current: Simplified mapping from velocity commands to thrust
   - Needed: Full geometric controller from aerial_gym

5. **Testing**: Verify physics match aerial_gym
   - Motor response times
   - Thrust-to-weight ratio
   - Collision detection
   - Reward magnitudes

### Differences from aerial_gym_simulator

| Aspect | aerial_gym_simulator | aerial_lab |
|--------|---------------------|------------|
| **Physics Engine** | Isaac Gym (PhysX 4) | Isaac Sim (PhysX 5) |
| **Scene Management** | Manual env cloning | InteractiveScene |
| **DR System** | Manual in reset_idx | EventManager + manual |
| **Rendering** | Warp (custom GPU raycast) | USD cameras |
| **Configuration** | Plain Python classes | @configclass decorators |
| **Observation Stacking** | Manual tensor management | Built-in history support |
| **Asset Loading** | Custom URDF loader | UrdfFileCfg |

## Directory Structure

```
aerial_lab/
├── __init__.py              # Gym registration
├── README.md                # This file
├── navigation_env.py        # Main environment implementation
├── navigation_env_cfg.py    # Configuration classes
├── lmf2/                    # Robot asset
│   └── model.urdf
└── agents/                  # RL agent configs
    ├── __init__.py
    └── rsl_rl_ppo_cfg.py   # PPO hyperparameters
```

## Citation

If you use this environment, please cite both IsaacLab and aerial_gym_simulator:

```bibtex
@article{mittal2023orbit,
  title={Orbit: A unified simulation framework for interactive robot learning environments},
  author={Mittal, Mayank and Yu, Calvin and Yu, Qinxi and Liu, Jingzhou and Rudin, Nikita and Hoeller, David and Yuan, Jia Lin and Singh, Ritvik and Guo, Yunrong and Mazhar, Hammad and others},
  journal={IEEE Robotics and Automation Letters},
  year={2023}
}

@inproceedings{kulkarni2023aerial,
  title={An open-source framework for parallelizing reinforcement learning for micro aerial vehicles},
  author={Kulkarni, Mihir and Li, Amogh and Dharmadhikari, Mihir and Dharmadhikari, Mudit and Alexis, Kostas},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2024}
}
```

## Troubleshooting

### Import Errors

If you get import errors, make sure IsaacLab is properly installed:

```bash
cd /path/to/IsaacLab
./isaaclab.sh --install
```

### Camera Not Working

Depth camera requires rendering. Don't use `--headless` during debugging:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Aerial-Navigation-Direct-v0
```

### Motor Dynamics Unstable

Try reducing time constants or increasing decimation:

```python
cfg.sim.dt = 1.0 / 200.0  # 200 Hz physics
cfg.decimation = 4  # 50 Hz control
```

## Contributing

To extend this environment:

1. **Add More DR**: Implement additional EventTerms in `EventCfg`
2. **Improve Controller**: Replace simplified controller with full Lee controller
3. **Add Obstacles**: Implement dynamic obstacle spawning based on curriculum
4. **Train VAE**: Train depth encoder and integrate
5. **Add Sensors**: IMU, LiDAR, additional cameras

## License

BSD-3-Clause (matching IsaacLab)

## Acknowledgments

- Original aerial_gym_simulator: NTNU-ARL
- IsaacLab framework: ETH Zurich RSL
- Conversion: Claude Code assistance
