# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Aerial Lab: Vision-based navigation environments for aerial robots.

Converted from aerial_gym_simulator with extensive domain randomization.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments
##

gym.register(
    id="Isaac-Aerial-Navigation-Direct-v0",
    entry_point="isaaclab_tasks.direct.aerial_lab.navigation_env:AerialNavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.aerial_lab.navigation_env_cfg:AerialNavigationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AerialNavigationPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Aerial-Navigation-Direct-v0-Play",
    entry_point="isaaclab_tasks.direct.aerial_lab.navigation_env:AerialNavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.aerial_lab.navigation_env_cfg:AerialNavigationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AerialNavigationPPORunnerCfg",
    },
)
