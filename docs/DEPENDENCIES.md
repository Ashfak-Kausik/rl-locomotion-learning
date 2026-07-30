# Dependency Checklist

Everything this project needs to build and run, why it needs it, and how to
get it. **The short version:**

```bash
./scripts/setup_env.sh          # detects the OS, installs everything, verifies
source .venv/bin/activate
python scripts/check_env.py     # re-verify any time
```

That script is idempotent — re-run it whenever something looks broken.

---

## Contents

1. [Quick start — three paths](#1-quick-start--three-paths)
2. [The complete checklist](#2-the-complete-checklist)
3. [System-level dependencies](#3-system-level-dependencies)
4. [Python packages](#4-python-packages)
5. [Data assets](#5-data-assets)
6. [The one dependency we cannot install](#6-the-one-dependency-we-cannot-install)
7. [What runs without it](#7-what-runs-without-it)
8. [The tooling: `setup_env.sh` and `check_env.py`](#8-the-tooling)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Quick start — three paths

### Path A — automatic (recommended)

```bash
git clone <repo> && cd rl-locomotion-learning
./scripts/setup_env.sh
source .venv/bin/activate
python scripts/check_env.py
```

Supports apt (Debian/Ubuntu/Pop!\_OS/Mint), dnf (Fedora/RHEL), pacman (Arch),
zypper (openSUSE) and Homebrew (macOS). Useful flags:

| Flag | Effect |
|---|---|
| `--check-only` | report what is missing, change nothing |
| `--yes` | non-interactive (CI, scripts) |
| `--no-system` | skip OS packages entirely (no `sudo` needed) |
| `--cuda` | install CUDA PyTorch instead of the CPU build |
| `--python 3.11` | choose the interpreter |

### Path B — Docker (most reproducible)

```bash
docker compose -f docker/compose.yaml build
docker compose -f docker/compose.yaml run --rm lab
```

Zero host dependencies beyond Docker itself. See [DOCKER.md](DOCKER.md).

### Path C — manual

Follow §3 and §4 below by hand.

---

## 2. The complete checklist

Tick these off and the whole pipeline runs.

**System**
- [ ] Python **3.10+** (3.12 verified) — every core library requires it
- [ ] `git`
- [ ] A C/C++ toolchain (`build-essential` / `gcc`+`make`) — builds Box2D
- [ ] Python headers (`python3-dev` / `python3-devel`)
- [ ] `python3-venv` (Debian family ships venv separately)
- [ ] OpenGL runtime (`libgl1` / `mesa-libGL`)
- [ ] GLFW (`libglfw3` / `glfw`) — **only** for the interactive viewer
- [ ] OSMesa (`libosmesa6`) or EGL (`libegl1`) — **only** for headless rendering
- [ ] `swig` — **only** if Box2D has no prebuilt wheel for your platform
- [ ] `ffmpeg` — **only** for encoding demo GIFs
- [ ] Docker + Compose v2 — **only** for Path B

**Python** (all via `requirements.txt`, plus `requirements-dev.txt` for tests)
- [ ] `numpy>=1.26,<3`
- [ ] `mujoco>=3.2,<4`
- [ ] `torch>=2.4,<3` (CPU build is enough and 12× smaller)
- [ ] `matplotlib>=3.8,<4`
- [ ] `imageio>=2.34,<3`
- [ ] `gymnasium[box2d]>=1.0,<2`
- [ ] `stable-baselines3>=2.4,<3`
- [ ] `tensorboard>=2.16,<3`
- [ ] `tqdm`, `rich`
- [ ] `pytest>=8` (dev only — `pip install -r requirements-dev.txt`)

**Assets**
- [x] Unitree Go2 MJCF + 16 meshes — **vendored in-repo**, nothing to do
- [x] 11 terrain scene XMLs — **committed**, regenerable
- [x] Experiment CSVs — **committed**
- [ ] `walk-these-ways` `.jit` checkpoints — **you must supply these** (§6)

**Hardware**
- [x] Any x86-64 CPU. 8 cores is comfortable; the experiments are CPU-bound
- [x] ~4 GB RAM
- [x] ~3 GB disk (venv ≈ 1.5 GB, Docker image ≈ 2 GB)
- [ ] GPU — **not required**. Only Stage 3 (MJX training) will want one

---

## 3. System-level dependencies

| Package (apt) | Fedora | Arch | Required for | Hard requirement? |
|---|---|---|---|---|
| `build-essential` | `gcc gcc-c++ make` | `base-devel` | compiling Box2D | yes |
| `python3-dev` | `python3-devel` | (in base) | compiling Box2D | yes |
| `python3-venv` | (bundled) | (bundled) | creating `.venv` | yes |
| `libgl1` | `mesa-libGL` | `libgl` | MuJoCo rendering | yes |
| `libglfw3` | `glfw` | `glfw` | interactive viewer window | viewer only |
| `libosmesa6` | `mesa-libOSMesa` | `mesa` | `MUJOCO_GL=osmesa` | headless render only |
| `libegl1` | `mesa-libEGL` | `mesa` | `MUJOCO_GL=egl` | headless GPU render only |
| `swig` | `swig` | `swig` | Box2D bindings | only if no wheel |
| `ffmpeg` | `ffmpeg` | `ffmpeg` | GIF/MP4 encoding | media only |
| `patchelf` | `patchelf` | `patchelf` | occasional native fixups | rarely |
| `git` | `git` | `git` | version control | yes |

Debian / Ubuntu / Pop!\_OS one-liner:

```bash
sudo apt-get update && sudo apt-get install -y \
  build-essential pkg-config python3-dev python3-venv python3-pip \
  libgl1 libglfw3 libegl1 libosmesa6 swig ffmpeg patchelf git
```

> **On modern Box2D:** `box2d==2.3.10` now ships manylinux wheels for CPython
> 3.12, so `swig` is often unnecessary. Keep it installed anyway — it costs
> nothing and rescues you on any platform without a wheel.

---

## 4. Python packages

Install into a **virtualenv**, never system-wide. Debian-family distros
enforce this (PEP 668 `externally-managed-environment`), and they are right to.

```bash
python3 -m venv .venv
source .venv/bin/activate

# CPU-only torch FIRST, so pip never resolves the 2.5 GB CUDA wheel
pip install "torch>=2.4,<3" --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

Versions verified working together on Python 3.12 / Pop!\_OS 24.04:

| Package | Verified | Role |
|---|---|---|
| numpy | 2.5.1 | array maths everywhere |
| mujoco | 3.11.0 | physics engine, all of Stage 2 |
| torch | 2.13.0+cpu | TorchScript inference only |
| matplotlib | 3.11.1 | `make_figures.py` |
| imageio | 2.37.4 | offscreen screenshots in `08_view_scene.py` |
| gymnasium | 1.3.0 | Stage 1 environments (`LunarLander-v3` needs ≥ 1.0) |
| stable-baselines3 | 2.9.0 | Stage 1 PPO |
| tensorboard | 2.21.0 | Stage 1 training curves |
| box2d | 2.3.10 | LunarLander physics |
| tqdm / rich | 4.70 / 15.0 | SB3 `progress_bar=True` |

### Why CPU-only torch

The project is explicitly CPU-developed. Stage 2 inference costs **< 2 ms per
step**, well inside the 20 ms a 50 Hz loop allows. The CUDA wheel is ~2.5 GB
versus ~190 MB and buys nothing here. Pass `--cuda` to `setup_env.sh` if you
plan to train locally.

---

## 5. Data assets

### Vendored — nothing to do

```
stage2-go2-mujoco-inference/scenes/go2_model/
├── go2.xml          Unitree Go2 MJCF, from mujoco_menagerie
└── *.obj            16 visual/collision meshes (~27 MB)
```

Commit `2810d2b` ("Add reproducible self-contained Go2 model") moved this
in-tree deliberately. Before that, scripts pointed at an external
`mujoco_menagerie` checkout — the same class of problem as the policy paths.
`go2.xml` declares `meshdir="."`, so it resolves its own meshes with no
symlinks and no external clone.

### Generated but committed

The 11 terrain scenes are produced by `generate_terrain_scenes.py` and
committed so the exact geometry used in the results is reproducible. You can
verify that guarantee:

```bash
python stage2-go2-mujoco-inference/experiments/generate_terrain_scenes.py
git diff --stat stage2-go2-mujoco-inference/scenes/    # must be empty
```

*(Verified byte-identical during this documentation work.)*

### Experiment data

`experiments/results/*.csv` (3 files, 75 trials total) plus
`EXPERIMENT_FINDINGS.md`. Committed, which is what lets `make_figures.py`
regenerate every data figure with **no policy weights and no simulation**.

---

## 6. The one dependency we cannot install

### `walk-these-ways` policy checkpoints

```
policies/walk-these-ways-go2/
├── body_latest.jit                  actor        (1, 2102) → (1, 12)
└── adaptation_module_latest.jit     RMA student  (1, 2100) → (1, 2)
```

**Why they are not here:** they are TorchScript exports of a policy trained in
NVIDIA Isaac Gym on GPU hardware. They are large binaries with no stable public
download, and they are not this project's output — they are its *input*.

**How to get them:**

1. Obtain the two files from the `walk-these-ways` Go2 training run this
   project used (the original author's checkpoint directory was
   `runs/gait-conditioned-agility/pretrain-go2/train/142238.667503/checkpoints`).
2. Or export your own from a `walk-these-ways` training run —
   upstream: <https://github.com/Improbable-AI/walk-these-ways>.
   The Go2 port is a community fork of that repo.
3. Or, once **Stage 3** lands, train a replacement with MJX /
   `mujoco_playground` and export it to TorchScript.

**Where to put them:**

```bash
# Option 1 — the default location
cp body_latest.jit adaptation_module_latest.jit \
   policies/walk-these-ways-go2/

# Option 2 — keep them anywhere and point at it
export GO2_POLICY_DIR=/absolute/path/to/checkpoints
```

**Verify:**

```bash
python stage2-go2-mujoco-inference/paths.py      # prints OK / MISS per file
python scripts/check_env.py                      # full report
```

Resolution happens in one place —
[`stage2-go2-mujoco-inference/paths.py`](../stage2-go2-mujoco-inference/paths.py) —
so nothing else needs editing. Scripts that need the weights and cannot find
them now raise a message that says exactly what is missing and what to do.

---

## 7. What runs without it

A genuinely large fraction of the repo. Do not let the missing weights stop
you from getting productive on day one.

### ✅ Works with zero policy weights

| Command | What you get |
|---|---|
| `python scripts/check_env.py` | full environment report |
| `python stage1-rl-fundamentals/01_cartpole_ppo.py` | trains PPO in ~2 min |
| `python stage1-rl-fundamentals/03_lunarlander_ppo.py` | ~15 min, CPU |
| `python stage1-rl-fundamentals/05_pendulum_ppo.py` | ~10 min, CPU |
| `python stage1-rl-fundamentals/02_cartpole_watch.py` | watch a trained agent |
| `tensorboard --logdir stage1-rl-fundamentals/tb_logs` | training curves |
| `python stage2-go2-mujoco-inference/01_hello_go2.py` | Go2 in the viewer |
| `python stage2-go2-mujoco-inference/02_inspect_go2.py` | model structure dump |
| `python stage2-go2-mujoco-inference/03_pose_go2.py` | PD-held standing pose |
| `python stage2-go2-mujoco-inference/04_build_obs_vector.py` | obs construction (superseded layout) |
| `.../experiments/generate_terrain_scenes.py` | regenerate all 11 scenes |
| `.../experiments/check_terrain_visual.py go2_stairs_08.xml` | inspect terrain |
| `.../experiments/make_figures.py` | **all 4 data figures** from the CSVs |
| `make test` | the full 93-test suite (~3 s) |
| `python stage3-go2-training/train.py --smoke` | 30 s end-to-end training run |
| `python stage3-go2-training/export.py --random` | contract-valid TorchScript export |

That covers the entire RL curriculum, all MuJoCo fundamentals, the scene
system, and the full figure pipeline.

### ⛔ Needs policy weights

`05_inspect_policy.py`, `06_run_policy.py`, `07_run_policy_interactive.py`,
`08_view_scene.py`, `experiments/harness.py`, `exp1_velocity_sweep.py`,
`exp2_gait_robustness.py`, `exp3_terrain.py`.

In other words: only the things that make the robot actually *walk*.

---

## 8. The tooling

### `scripts/check_env.py` — verify

Read-only. Checks four layers and exits non-zero if a **required** item is
missing (optional items are `WARN` and never fail the run).

```
1. SYSTEM   OS, Python version, venv, native libs, Docker, GPU
2. PYTHON   every package + version, plus which torch build is installed
3. ASSETS   MJCF, meshes, terrain scenes, policy weights, result CSVs
4. RUNTIME  does MuJoCo really load go2_flat.xml? do the Gym envs build?
```

```bash
python scripts/check_env.py              # full report
python scripts/check_env.py --quiet      # problems only
python scripts/check_env.py --json       # machine-readable, for CI
```

Layer 4 is the valuable part — it catches the failures that a package-version
check cannot, like a MuJoCo install that imports fine but cannot parse a model.

### `scripts/setup_env.sh` — install

Five idempotent steps: detect platform → install native packages → pick a
Python → create `.venv` and install packages → run `check_env.py`.

It never installs native packages without asking, unless you pass `--yes`.

---

## 9. Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `error: externally-managed-environment` | PEP 668 blocks system pip | use the venv: `./scripts/setup_env.sh` |
| `ModuleNotFoundError: No module named 'mujoco'` | venv not activated | `source .venv/bin/activate` |
| `ERROR: Failed building wheel for box2d-py` | no toolchain | install `swig build-essential python3-dev`, retry |
| `ValueError: ... body_latest.jit does not exist` | no policy weights | §6 |
| `Error opening file '.../base_0.obj'` | relative path to `from_xml_path` | pass an absolute path |
| `GLFWError: X11: The DISPLAY environment variable is missing` | headless machine | `export MUJOCO_GL=osmesa` |
| Viewer opens black / crashes | GLFW or GL driver missing | install `libglfw3 libgl1` |
| `GLX: Failed to create context: BadValue` | GPU driver/library mismatch — usually an NVIDIA update with no reboot since | reboot. Immediate workaround: `LIBGL_ALWAYS_SOFTWARE=1 __GLX_VENDOR_LIBRARY_NAME=mesa python …` (software rendering, slower but correct) |
| `Image width N > framebuffer width 640` | MuJoCo's offscreen framebuffer defaults to 640×480 | set `model.vis.global_.offwidth/offheight` before constructing `mujoco.Renderer` |
| `nvidia-smi: Driver/library version mismatch` | driver updated without reboot | reboot — or ignore, Stages 1–2 are CPU-only |
| `python3 -m venv` fails | venv split into its own package | `sudo apt install python3-venv` |
| Docker files owned by root | uid mismatch | `UID=$(id -u) GID=$(id -g) docker compose ... build` |
| pip downloads a 2.5 GB torch | CUDA wheel resolved | install torch first from the CPU index (§4) |

Still stuck? `python scripts/check_env.py` prints exactly which layer failed
and why.
