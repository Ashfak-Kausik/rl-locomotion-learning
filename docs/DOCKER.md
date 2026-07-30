# Running in Docker

The container path gives you a byte-identical environment on any machine with
Docker installed — no Python, no system libraries, no version drift. This is
the recommended way to run experiments and the only sane way to reproduce
results across machines.

---

## TL;DR

```bash
# build once (~5 min, ~2 GB image)
UID=$(id -u) GID=$(id -g) docker compose -f docker/compose.yaml build

# interactive shell
docker compose -f docker/compose.yaml run --rm lab

# one-off command
docker compose -f docker/compose.yaml run --rm headless \
    python stage2-go2-mujoco-inference/experiments/make_figures.py

# GUI viewer (Linux desktop)
xhost +local:docker
docker compose -f docker/compose.yaml run --rm viewer
```

---

## What is in the image

| | |
|---|---|
| Base | `python:3.12-slim-bookworm` |
| Python deps | everything in `requirements.txt`, CPU-only PyTorch |
| Native libs | GL, GLFW, EGL, OSMesa, X11 client libs, ffmpeg, swig, build tools |
| Default renderer | `MUJOCO_GL=osmesa` — software, no GPU, no display needed |
| User | non-root, uid/gid matched to the host at build time |
| Size | ~2 GB (CUDA torch would make it ~5 GB — deliberately avoided) |

**CPU-only by design.** Stages 1 and 2 never need a GPU: policy inference is
< 2 ms/step on CPU. Stage 3 (MJX training) is GPU work and is expected to run
on a cloud notebook, not in this image.

---

## The four services

All four share one image and one bind-mounted repo. They differ only in how
they render.

### `lab` — interactive shell

```bash
docker compose -f docker/compose.yaml run --rm lab
```

Drops you at a `bash` prompt in `/workspace` with a banner showing your
Python/MuJoCo/torch versions and whether the policy weights are visible. Best
for exploration.

### `headless` — batch runner

```bash
docker compose -f docker/compose.yaml run --rm headless <command>
```

No display at all, quiet banner. This is the experiment workhorse:

```bash
# regenerate every figure from the committed CSVs (needs NO policy weights)
docker compose -f docker/compose.yaml run --rm headless \
    python stage2-go2-mujoco-inference/experiments/make_figures.py

# full experiment sweep (needs policy weights)
docker compose -f docker/compose.yaml run --rm headless \
    python stage2-go2-mujoco-inference/experiments/exp1_velocity_sweep.py

# environment report
docker compose -f docker/compose.yaml run --rm headless
```

### `viewer` — MuJoCo GUI through the host X server

```bash
xhost +local:docker      # once per login session
docker compose -f docker/compose.yaml run --rm viewer
```

Mounts `/tmp/.X11-unix` and sets `MUJOCO_GL=glfw` + `DISPLAY`. Works on X11 and,
via XWayland, on Wayland. Override the command to pick a script:

```bash
docker compose -f docker/compose.yaml run --rm viewer \
    python stage2-go2-mujoco-inference/07_run_policy_interactive.py
```

Revoke access when finished: `xhost -local:docker`.

### `tensorboard` — training curves

```bash
docker compose -f docker/compose.yaml up tensorboard
# → http://localhost:6006
```

Change the port with `TB_PORT=6007 docker compose ... up tensorboard`.

---

## Volumes and paths

```
  HOST                                CONTAINER
  ./                          ──rw──► /workspace          the repo, live-mounted
  ${GO2_POLICY_SRC:-./policies} ─ro─► /workspace/policies  the .jit checkpoints
  /tmp/.X11-unix              ──rw──► /tmp/.X11-unix      (viewer service only)
```

The repo is **bind-mounted, not copied**. Edit a file on the host and the
change is live in the container immediately — no rebuild. Results written by
the container (CSVs, figures) land straight in your working tree.

Policy checkpoints are mounted **read-only** from outside the repo:

```bash
# keep the weights anywhere on the host
GO2_POLICY_SRC=/data/models/wtw docker compose -f docker/compose.yaml run --rm lab
```

Inside the container, `GO2_POLICY_DIR=/workspace/policies/walk-these-ways-go2`
is preset, and `paths.py` reads it. See [DEPENDENCIES.md §6](DEPENDENCIES.md#6-the-one-dependency-we-cannot-install).

---

## File ownership

Pass your uid/gid at build time so container-written files belong to you:

```bash
UID=$(id -u) GID=$(id -g) docker compose -f docker/compose.yaml build
```

Skip this and files may come out owned by uid 1000 (harmless if that is you,
annoying otherwise). If you already have root-owned artefacts:

```bash
sudo chown -R "$(id -u):$(id -g)" stage2-go2-mujoco-inference/experiments/
```

---

## Rendering backends

`MUJOCO_GL` decides how MuJoCo draws. The compose services set it for you.

| Value | Needs | Speed | Used by |
|---|---|---|---|
| `osmesa` | nothing | slow (CPU) | `lab`, `headless` |
| `egl` | GPU + drivers in-container | fast | manual override |
| `glfw` | X server / display | fast | `viewer` |

Experiments do **no rendering at all**, so `MUJOCO_GL` is irrelevant to their
speed — `harness.py` never constructs a renderer.

---

## Resource limits

`mem_limit` defaults to 8 GB so a runaway experiment cannot OOM the host:

```bash
RLL_MEM_LIMIT=16g docker compose -f docker/compose.yaml run --rm headless ...
```

Experiments are CPU-bound and single-threaded per trial. To pin cores:

```bash
docker compose -f docker/compose.yaml run --rm --cpus 4 headless ...
```

---

## Rebuilding

| Change | Action |
|---|---|
| Python code | nothing — the repo is bind-mounted |
| `requirements.txt` | `docker compose -f docker/compose.yaml build` |
| `Dockerfile` | same |
| Force a clean build | add `--no-cache` |

Dependency layers are cached independently of the code, because
`requirements.txt` is copied before anything else. Editing a `.py` file never
triggers a pip reinstall.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Authorization required, but no authorization protocol specified` | run `xhost +local:docker` on the host |
| Viewer service exits instantly | no display — use `lab`/`headless` instead |
| `permission denied` writing results | rebuild with `UID=$(id -u) GID=$(id -g)` |
| `body_latest.jit does not exist` | mount the weights: `GO2_POLICY_SRC=/path/to/dir` |
| Build fails downloading torch | network/proxy; retry, or build with `--network host` |
| Image is enormous | you built with CUDA torch — the CPU index is the default here |
| `docker compose` not found | Compose v2 plugin missing: `sudo apt install docker-compose-plugin` |

Inside any container, `python scripts/check_env.py` tells you exactly what the
container can and cannot see.

---

## Container vs. host virtualenv

| | Docker | `.venv` |
|---|---|---|
| Reproducibility | identical everywhere | depends on host libs |
| Setup | one build | `./scripts/setup_env.sh` |
| Interactive viewer | needs X11 plumbing | native, just works |
| Experiment runs | ideal | fine |
| Editing/debugging | fine (bind mount) | slightly smoother |
| Disk | ~2 GB image | ~1.5 GB venv |

Reasonable habit: **develop in the venv, run experiments and publish results
from Docker.**
