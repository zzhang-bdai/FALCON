# ============================================================
# FALCON - Isaac Gym Preview 4 Training Container
# Base: CUDA 11.8 runtime on Ubuntu 20.04 + Miniconda
# ============================================================
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# ---- Non-root user (created early so installs are owned by devuser) ----
ARG HOST_UID=1000
ARG HOST_GID=1000
RUN apt-get update && apt-get install -y --no-install-recommends sudo && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd -g ${HOST_GID} devuser && \
    useradd -m -s /bin/bash -u ${HOST_UID} -g ${HOST_GID} -G sudo devuser --no-log-init && \
    echo "devuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# ---- System dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Rendering (EGL/Vulkan for headless, X11/GLX for GUI)
    libvulkan1 vulkan-utils libgl1-mesa-glx libgl1-mesa-dev libgles2-mesa \
    libegl1-mesa libegl1-mesa-dev \
    libglfw3 libglfw3-dev \
    libxrandr2 libxinerama1 libxcursor1 libxi6 \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    # Build essentials
    build-essential cmake git wget curl \
    # Open3D and misc system deps
    libusb-1.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ---- Install Miniforge + create fcgym & fcreal envs (single layer to reduce export size) ----
USER devuser
ENV CONDA_DIR=/home/devuser/conda
RUN wget -qO /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh && \
    ${CONDA_DIR}/bin/conda create -n fcgym python=3.8 -y && \
    ${CONDA_DIR}/bin/conda create -n fcreal python=3.10 -y && \
    ${CONDA_DIR}/bin/conda clean -afy
ENV PATH=${CONDA_DIR}/bin:${PATH}

# Make all subsequent RUN commands use the fcgym environment
SHELL ["conda", "run", "-n", "fcgym", "/bin/bash", "-c"]

# ---- Vulkan ICD configuration for GPU rendering ----
USER root
RUN mkdir -p /usr/share/vulkan/icd.d
COPY docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json

# ---- Environment variables ----
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---- Working directory ----
RUN mkdir -p /workspace && chown devuser:devuser /workspace
WORKDIR /workspace
USER devuser

# ---- Install PyTorch (matching Isaac Gym Preview 4) ----
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# ---- Install FALCON and isaac_utils in fcgym (caches dependencies) ----
COPY --chown=devuser:devuser setup.py /workspace/setup.py
COPY --chown=devuser:devuser isaac_utils/setup.py /workspace/isaac_utils/setup.py
RUN mkdir -p /workspace/isaac_utils/isaac_utils && \
    touch /workspace/isaac_utils/isaac_utils/__init__.py && \
    pip install -e /workspace -e /workspace/isaac_utils

# ---- Shell init ----
RUN conda init bash

# ---- Install fcreal environment packages ----
SHELL ["conda", "run", "-n", "fcreal", "/bin/bash", "-c"]
RUN conda install pinocchio=3.2.0 --override-channels -c conda-forge -y && \
    conda clean -afy
RUN git clone https://github.com/unitreerobotics/unitree_sdk2_python.git /home/devuser/unitree_sdk2_python && \
    pip install -e /home/devuser/unitree_sdk2_python
COPY --chown=devuser:devuser sim2real/requirements.txt /workspace/sim2real/requirements.txt
RUN pip install -r /workspace/sim2real/requirements.txt

# ---- Install Claude Code ----
RUN curl -fsSL https://claude.ai/install.sh | bash
RUN echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc

# ---- Entrypoint script ----
SHELL ["/bin/bash", "-c"]
USER root
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER devuser

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
