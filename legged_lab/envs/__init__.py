# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg

from legged_lab.envs.g1.g1_locomotion_env import G1LocomotionEnv
from legged_lab.envs.g1.g1_locomotion_cfg import G1LocomotionAgentCfg, G1LocomotionEnvCfg

from legged_lab.envs.g1.g1_navigation_env import G1NavigationEnv
from legged_lab.envs.g1.g1_navigation_cfg import G1NavigationAgentCfg, G1NavigationEnvCfg

from legged_lab.utils.task_registry import task_registry

task_registry.register("g1_locomotion", G1LocomotionEnv, G1LocomotionEnvCfg(), G1LocomotionAgentCfg())
task_registry.register("g1_navigation", G1NavigationEnv, G1NavigationEnvCfg(), G1NavigationAgentCfg())
