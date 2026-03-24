# ============================================================
# FALCON - Isaac Gym Preview 4 Training Container
# Base: CUDA 11.8 devel on Ubuntu 20.04 + Miniconda
# ============================================================
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# ---- Non-root user (created early so installs are owned by devuser) ----
ARG HOST_UID=1000
ARG HOST_GID=1000
RUN apt-get update && apt-get install -y --no-install-recommends sudo && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd -g ${HOST_GID} devuser && \
    useradd -m -s /bin/bash -u ${HOST_UID} -g ${HOST_GID} -G sudo devuser && \
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

# ---- Install Miniconda + create fcgym env (single layer to reduce export size) ----
USER devuser
ENV CONDA_DIR=/home/devuser/conda
RUN wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniconda.sh && \
    ${CONDA_DIR}/bin/conda create -n fcgym python=3.8 -y --override-channels -c conda-forge && \
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

# ---- Shell init ----
RUN conda init bash

# ---- Entrypoint script ----
USER root
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

SHELL ["/bin/bash", "-c"]
USER devuser

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
