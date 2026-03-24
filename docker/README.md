# FALCON Docker Setup

Docker container for FALCON training with Isaac Gym Preview 4 on Ubuntu 20.04 + CUDA 11.8.

## Prerequisites

1. **NVIDIA Container Toolkit** installed on host:
   ```bash
   # Install (if not already)
   sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

2. **Isaac Gym Preview 4** downloaded and extracted at `../isaacgym` (sibling to FALCON):
   ```bash
   # Download from https://developer.nvidia.com/isaac-gym/download
   cd /home/zzhang/Dev
   tar -xzf IsaacGym_Preview_4_Package.tar.gz
   ```

   Expected directory structure:
   ```
   Dev/
   ├── FALCON/        # This repo
   └── isaacgym/      # Isaac Gym Preview 4
       └── python/
   ```

## Usage

### Build the image

```bash
docker compose build
```

### Interactive session

```bash
# Allow X11 access for GUI rendering
xhost +local:docker

# Start interactive bash
docker compose run --rm falcon bash
```

### Run training

```bash
# Headless training
docker compose run --rm falcon python humanoidverse/train_agent.py \
  +exp=decoupled_locomotion_stand_height_waist_wbc_diff_force_ma_ppo_ma_env \
  +simulator=isaacgym \
  +domain_rand=domain_rand_rl_gym \
  +rewards=dec_loco/reward_dec_loco_stand_height_ma_diff_force \
  +robot=g1/g1_29dof_waist_fakehand \
  +terrain=terrain_locomotion_plane \
  +obs=dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma \
  num_envs=4096 \
  project_name=g1_29dof_falcon \
  experiment_name=g1_29dof_falcon

# With GUI viewer (requires xhost +local:docker)
# Remove --headless flag or it's not set by default
```

### Run evaluation

```bash
docker compose run --rm falcon python humanoidverse/eval_agent.py \
  +checkpoint=/workspace/outputs/your_checkpoint.pt
```

### Verify setup

```bash
# Check GPU access
docker compose run --rm falcon nvidia-smi

# Check Isaac Gym import
docker compose run --rm falcon python -c "import isaacgym; print('Isaac Gym OK')"

# Check PyTorch CUDA
docker compose run --rm falcon python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## W&B Integration

```bash
export WANDB_API_KEY=your_key_here
export WANDB_MODE=online  # default is disabled

docker compose run --rm falcon python humanoidverse/train_agent.py \
  ... +opt=wandb
```

## Troubleshooting

- **No GUI window**: Run `xhost +local:docker` on the host before starting the container
- **libpython error**: The entrypoint sets `LD_LIBRARY_PATH` automatically
- **Isaac Gym not found**: Ensure `../isaacgym/python` exists relative to this repo
- **Permission denied on X11**: Check `echo $DISPLAY` returns a value (e.g., `:1`)
