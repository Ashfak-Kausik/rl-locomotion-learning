# =============================================================================
# rl-locomotion-learning — reproducible CPU-only image
# =============================================================================
# Deliberately CPU-only: Stages 1 and 2 of this project never need a GPU, and
# a CPU torch wheel is ~200 MB versus ~2.5 GB for the CUDA one. Stage 3
# (MJX / mujoco_playground training) is GPU work and is expected to run on a
# cloud notebook, not in this image.
#
# Build:  docker compose -f docker/compose.yaml build
# Run:    docker compose -f docker/compose.yaml run --rm lab
# =============================================================================
FROM python:3.12-slim-bookworm

# --- OS layer ---------------------------------------------------------------
#   libgl1 / libglfw3      MuJoCo rendering + interactive viewer window
#   libegl1 / libosmesa6   headless rendering backends (MUJOCO_GL)
#   libx11/xext/xrandr/..  X11 client libs, so the viewer can reach a host
#                          X server via a mounted socket
#   build-essential+swig   compile Box2D for LunarLander-v3
#   ffmpeg                 encode demo GIFs / MP4s
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        swig \
        git \
        ffmpeg \
        patchelf \
        libgl1 \
        libglfw3 \
        libegl1 \
        libosmesa6 \
        libx11-6 \
        libxext6 \
        libxrandr2 \
        libxinerama1 \
        libxcursor1 \
        libxi6 \
        libxxf86vm1 \
    && rm -rf /var/lib/apt/lists/*

# --- Python layer -----------------------------------------------------------
# Copy requirements first so dependency layers cache independently of code.
WORKDIR /workspace
COPY requirements.txt /tmp/requirements.txt

# CPU-only torch from PyTorch's dedicated index, installed BEFORE the rest so
# pip never resolves the CUDA variant as a transitive dependency.
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir "torch>=2.4,<3" \
        --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt

# --- Runtime configuration --------------------------------------------------
# MUJOCO_GL=osmesa  -> pure software rendering, works with zero GPU and zero
#                      X server. Overridden to "glfw" by the `viewer` service
#                      in compose.yaml, which does have an X socket.
# PYTHONUNBUFFERED  -> experiment progress prints stream out live.
ENV MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    GO2_POLICY_DIR=/workspace/policies/walk-these-ways-go2

# Run as a non-root user whose uid/gid can be overridden at build time so
# files written into the bind-mounted repo stay owned by the host user.
ARG UID=1000
ARG GID=1000
RUN groupadd -g "${GID}" lab 2>/dev/null || true \
    && useradd -m -u "${UID}" -g "${GID}" -s /bin/bash lab 2>/dev/null || true \
    && mkdir -p /workspace && chown -R "${UID}:${GID}" /workspace

COPY --chown=${UID}:${GID} docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

USER ${UID}:${GID}
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
