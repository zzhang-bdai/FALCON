# ============================================================
# FALCON - Isaac Gym Preview 4 Training Container
# Base: CUDA 11.8 devel on Ubuntu 20.04
# ============================================================
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# ---- System dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python build
    python3.8 python3.8-dev python3.8-venv python3-pip \
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

# Make python3.8 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    python -m pip install --upgrade pip setuptools wheel

# ---- Vulkan ICD configuration for GPU rendering ----
RUN mkdir -p /usr/share/vulkan/icd.d
COPY docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json

# ---- Environment variables ----
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---- Working directory ----
WORKDIR /workspace

# ---- Install PyTorch (matching Isaac Gym Preview 4) ----
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# ---- Install FALCON Python dependencies ----
# Copy only dependency-defining files first for Docker layer caching
COPY setup.py /workspace/setup.py
COPY isaac_utils/setup.py /workspace/isaac_utils/setup.py

# Create minimal package structure so pip install -e works
RUN mkdir -p /workspace/humanoidverse /workspace/isaac_utils/isaac_utils && \
    touch /workspace/humanoidverse/__init__.py && \
    touch /workspace/isaac_utils/isaac_utils/__init__.py

RUN pip install -e /workspace && \
    pip install -e /workspace/isaac_utils

# ---- Entrypoint script ----
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
