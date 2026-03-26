#!/bin/bash
set -e

# Fix ownership of bind-mounted directories when running as non-root
if [ "$(id -u)" -ne 0 ]; then
    for dir in /workspace/logs /workspace/runs /workspace/outputs /workspace/wandb; do
        sudo mkdir -p "$dir"
        sudo chown -R "$(id -u):$(id -g)" "$dir"
    done
fi

# Activate the fcgym conda environment
eval "$(${CONDA_DIR}/bin/conda shell.bash hook)"
conda activate fcgym

# Install isaacgym from volume mount (only on first run)
STAMP="/home/devuser/.isaacgym_installed"
if [ -d "/workspace/isaacgym/python" ] && [ ! -f "$STAMP" ]; then
    pip install -e /workspace/isaacgym/python
    touch "$STAMP"
fi

# Set LD_LIBRARY_PATH for isaacgym native libs and conda libs
if [ -d "/workspace/isaacgym/python/isaacgym/_bindings/linux-x86_64" ]; then
    echo 'export LD_LIBRARY_PATH=/workspace/isaacgym/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH}' >> ~/.bashrc && source ~/.bashrc
fi
echo `export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}` >> ~/.bashrc && source ~/.bashrc

exec "$@"
