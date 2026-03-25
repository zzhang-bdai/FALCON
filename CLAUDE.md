# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FALCON (Learning Force-Adaptive Humanoid Loco-Manipulation) is a reinforcement learning framework for training humanoid robots using a **dual-agent decoupled control architecture** -- separate PPO policies for lower body (locomotion, 15 DoF) and upper body (manipulation, 14 DoF). Supports Unitree G1 and Booster T1 robots. Built on IsaacGym Preview 4 with Hydra configuration.

## Commands

### Installation
```bash
# Docker (recommended)
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose build --no-cache falcon
docker compose run --rm falcon bash

# Conda
conda create -n fcgym python=3.8 && conda activate fcgym
pip install -e isaacgym/python   # IsaacGym Preview 4 must be downloaded separately
pip install -e .
pip install -e isaac_utils
```

### Training (G1 example)
```bash
python humanoidverse/train_agent.py \
  +exp=decoupled_locomotion_stand_height_waist_wbc_diff_force_ma_ppo_ma_env \
  +simulator=isaacgym +domain_rand=domain_rand_rl_gym \
  +rewards=dec_loco/reward_dec_loco_stand_height_ma_diff_force \
  +robot=g1/g1_29dof_waist_fakehand \
  +terrain=terrain_locomotion_plane \
  +obs=dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma \
  num_envs=4096 project_name=g1_29dof_falcon experiment_name=g1_29dof_falcon
```
Add `+opt=wandb` for W&B logging. Training converges around 6k iterations.

### Evaluation & ONNX Export
```bash
python humanoidverse/eval_agent.py +checkpoint=<path_to_ckpt>
```
Automatically exports ONNX to `<checkpoint_dir>/exported/`.

### Sim2Sim Deployment (from `sim2real/` directory)
```bash
# Terminal 1: MuJoCo env
python sim_env/loco_manip.py --config=config/g1/g1_29dof_falcon.yaml
# Terminal 2: Policy
python rl_policy/loco_manip/loco_manip.py --config=config/g1/g1_29dof_falcon.yaml --model_path=models/falcon/g1_29dof.onnx
```

## Architecture

### Configuration System (Hydra)
All config lives in `humanoidverse/config/` with modular YAML composition. Key config groups:
- `exp/` -- complete experiment presets
- `algo/` -- PPO hyperparameters and network architecture
- `env/` -- task/environment setup (forces, commands, termination)
- `obs/` -- observation features and history configuration
- `rewards/` -- reward function definitions and scales (40+ components)
- `robot/` -- per-robot joint limits, DoF mapping, body separation
- `simulator/` -- physics engine settings
- `domain_rand/` -- domain randomization parameters
- `terrain/` -- terrain generation

Base config: `humanoidverse/config/base.yaml`. Override via CLI args (e.g., `num_envs=64`).

### Training Pipeline
`humanoidverse/train_agent.py` -> Hydra config -> simulator init -> `BaseTask` env (via `instantiate()`) -> `PPOMultiActorCritic` algo -> training loop.

Outputs to `logs/{project}/{timestamp}-{experiment}/` with checkpoints every 100 iterations.

### Core Components
- **`humanoidverse/agents/`** -- RL algorithms. `decouple/` has `PPOMultiActorCritic` for the dual-agent architecture. `modules/` has actor/critic/encoder networks.
- **`humanoidverse/envs/`** -- Task environments. `decoupled_locomotion/` is the main FALCON task with force application on end-effectors. `base_task/` provides the abstract interface.
- **`humanoidverse/simulator/`** -- Physics backends (IsaacGym primary; IsaacSim and Genesis in development).
- **`humanoidverse/data/`** -- Robot URDF/MJCF assets, motion capture data, object models.
- **`sim2real/`** -- Deployment stack: ONNX policy runners, MuJoCo sim environment, robot SDK wrappers (Unitree/Booster), IK via Pinocchio.
- **`isaac_utils/`** -- Shared utilities for IsaacGym/IsaacSim tensor operations.

### Import Order Constraint
`isaacgym` must be imported before `torch`. Both `train_agent.py` and `eval_agent.py` handle this by importing torch after the simulator check.

### Sim2Real has a separate conda env
The `sim2real/` deployment code uses Python 3.10 (`fcreal` env) with Pinocchio 3.2.0 for IK, separate from the training env (Python 3.8, `fcgym`).

## Key Patterns

- Components are composed via Hydra `instantiate()` -- the `_target_` field in YAML configs points to Python classes
- Logging uses `loguru` (not stdlib logging). Console level controlled by `LOGURU_LEVEL` env var
- No formal test suite; validation is through training runs and sim2sim deployment
- Reward functions are defined in YAML with configurable scales; implementation in environment classes
- Observation space is fully YAML-driven with automatic dimension calculation and optional history/noise
