# Dependency Checklist

Everything this project needs to build and run, why it needs it, and how to
get it. **The short version:**

```bash
./scripts/setup_env.sh          # detects the OS, installs everything, verifies
source .venv/bin/activate
python scripts/check_env.py     # re-verify any time
```

That script is idempotent — re-run it whenever something looks broken.
It is the **standalone** bootstrap: detects the OS, installs missing system
packages, creates `.venv`, installs the right PyTorch wheel for your GPU
(or CPU), then verifies with `check_env.py`. You do not need Docker, CI, or a
pre-configured machine image.

For Stage 3 training on non-NVIDIA GPUs (Intel Arc, AMD ROCm, Apple MPS), see
[§4.1 GPU flavours](#41-gpu-flavours--nvidia-intel-arc-amd-apple).

---

## Contents

1. [Quick start — three paths](#1-quick-start--three-paths)
2. [The complete checklist](#2-the-complete-checklist)
3. [System-level dependencies](#3-system-level-dependencies)
4. [Python packages](#4-python-packages)
5. [Data assets](#5-data-assets)
6. [The one dependency we cannot install](#6-the-one-dependency-we-cannot-install)
7. [What runs without it](#7-what-runs-without-it)
8. [The tooling: `setup_env.sh`, `check_env.py`, `hw_profile.py`](#8-the-tooling)
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
| `--gpu auto` | detect NVIDIA / Intel Arc / AMD / Apple and install the matching torch wheel |
| `--cuda` | NVIDIA CUDA PyTorch (pinned to the `cu130` index this repo verifies) |
| `--xpu` / `--arc` | Intel Arc XPU PyTorch |
| `--rocm` | AMD ROCm PyTorch |
| `--gpu cpu` | CPU-only PyTorch (default — Stages 1–2 need nothing else) |
| `--python 3.11` | choose the interpreter |

After install, record the machine so training results are comparable across
contributors:

```bash
make hw-profile          # writes hardware/profiles/<hostname>.json
make hw-check            # does this host match a known profile?
make check               # 4-layer dependency report
```

### Path B — manual

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
- [ ] GPU — **not required** for Stages 1–2. Stage 3 training wants one; see §4.1 for NVIDIA / Arc / ROCm / MPS

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

### Why CPU-only torch by default

The project is explicitly CPU-developed for Stages 1–2. Stage 2 inference
costs **< 2 ms per step**, well inside the 20 ms a 50 Hz loop allows. The CUDA
wheel is ~2.5 GB versus ~190 MB and buys nothing for inference. Pass
`--cuda` / `--xpu` / `--rocm` / `--gpu auto` only when you plan to **train**
(Stage 3) on that machine.

### 4.1 GPU flavours — NVIDIA, Intel Arc, AMD, Apple

Stages 1–2 never need a GPU. Stage 3 training does, and the install path
depends on the vendor. The bootstrap chooses the wheel; `check_env.py` reports
every vendor it can see; `tune_profiles.py` picks batch sizes by backend + VRAM.

| Your hardware | Bootstrap | What to expect | Train with |
|---|---|---|---|
| **No GPU / unknown** | `./scripts/setup_env.sh` (default CPU) | Stages 1–2 fully; Stage 3 smoke works, full train is slow | `--device cpu` |
| **NVIDIA** (GeForce / RTX) | `./scripts/setup_env.sh --cuda` or `--gpu auto` | Verified path on this repo (RTX 3050 8 GB measured). Needs a working driver; if `nvidia-smi` fails after an upgrade, **reboot** | `--device auto` (or `cuda`) |
| **Intel Arc** (A770 / A750 / Battlemage, …) | `./scripts/setup_env.sh --xpu` | Installs the XPU wheel from PyTorch's `xpu` index. `resolve_device()` uses `torch.xpu`. Profiles `arc-8gb` / `arc-16gb` are **unbenchmarked** starting points — run `hw_profile.py --save` and send measured sps back | `--device auto` (or `xpu`) |
| **AMD** (ROCm-capable) | `./scripts/setup_env.sh --rocm` | Installs the ROCm wheel. PyTorch exposes it via the CUDA API (`torch.cuda.is_available()` True under HIP). No measured ROCm profile yet — VRAM-based `cuda-*` profiles are used as a conservative stand-in; treat sps as unknown until you measure | `--device auto` |
| **Apple Silicon** | `./scripts/setup_env.sh` (CPU index; MPS ships in the macOS wheel) | `torch.backends.mps` when available; profile `apple-mps` is unbenchmarked. MuJoCo viewer/OpenGL on macOS has its own quirks — prefer offscreen for demos | `--device auto` (or `mps`) |

**Suggested first-run checklist on a new GPU machine:**

```bash
./scripts/setup_env.sh --gpu auto          # or --cuda / --xpu / --rocm
source .venv/bin/activate
python scripts/check_env.py                # must say READY; GPU line should be OK/WARN with a name
python scripts/hw_profile.py --save        # commit hardware/profiles/<host>.json
python scripts/gpu_check.py                # NVIDIA-focused; skip on Arc/AMD if N/A
python stage3-go2-training/train.py --smoke
python stage3-go2-training/train.py --run-name probe --timesteps 200000
# read steps/s from the log; if far from the profile rationale, open an issue
# or PR updating stage3-go2-training/tune_profiles.py
```

If `check_env.py` reports Arc hardware but `torch.xpu` unavailable, the fix is
exactly `./scripts/setup_env.sh --xpu` (reinstall torch from the XPU index) —
not a driver reboot. The opposite is usually true on NVIDIA: wheel is fine,
driver/library mismatch needs a reboot (`make gpu` spells this out).

Silent CPU fallback is the failure mode this tooling exists to prevent. With
`--device auto`, `resolve_device()` prints **why** it picked CPU when a GPU
was requested but unavailable.

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

## 6. The policy weights are not committed here, but ARE downloadable

### `walk-these-ways` policy checkpoints

```
policies/walk-these-ways-go2/
├── body_latest.jit                  actor        (1, 2102) → (1, 12)
└── adaptation_module_latest.jit     RMA student  (1, 2100) → (1, 2)
```

**Why they are not committed to THIS repo:** they are TorchScript exports of a
policy trained in NVIDIA Isaac Gym on GPU hardware — a large binary and not
this project's output, just its input. `policies/` is gitignored on purpose
(see `.gitignore`).

**They previously read as unobtainable here. They are not.** The Go2 fork
commits its pretrained checkpoint straight into git, MIT licensed:
<https://github.com/Teddy-Liao/walk-these-ways-go2>. Verified 2026-07-31 —
downloaded, loaded with `torch.jit.load`, and confirmed to satisfy the 70-dim
contract exactly (`(1,2100)→(1,2)`, `(1,2102)→(1,12)`) before trusting them.

```bash
mkdir -p policies/walk-these-ways-go2
BASE="https://raw.githubusercontent.com/Teddy-Liao/walk-these-ways-go2/main/runs/gait-conditioned-agility/pretrain-go2/train/142238.667503/checkpoints"
curl -sL --fail "$BASE/body_latest.jit"              -o policies/walk-these-ways-go2/body_latest.jit
curl -sL --fail "$BASE/adaptation_module_latest.jit" -o policies/walk-these-ways-go2/adaptation_module_latest.jit
```

Other ways to get the same two files, if that mirror ever moves:

1. Upstream `walk-these-ways` (Go1, not Go2) ships its own pretrained run —
   <https://github.com/Improbable-AI/walk-these-ways>, `runs/pretrain-v0/`.
   Not contract-compatible with this repo (different robot); useful only as
   a reference for where checkpoints live in this family of repos.
2. Export your own from a `walk-these-ways-go2` training run.
3. Train a replacement in **Stage 3** (`stage3-go2-training/`) and export it
   with `export.py` — no GPU compute currently available for a converged run,
   but the pipeline is complete and produces contract-valid weights today
   (`--smoke`, or `--random` for an untrained control condition).

**Where to put them:**

```bash
# Option 1 — the default location (matches the curl commands above)
# already correct if you used them as shown

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

## 8. The tooling

### `scripts/setup_env.sh` — install (standalone bootstrap)

Five idempotent steps: detect platform → install native packages → pick a
Python → create `.venv` and install packages (torch from the flavour-specific
index) → run `check_env.py`.

It never installs native packages without asking, unless you pass `--yes`.
Default torch is CPU; GPU wheels are opt-in via `--cuda` / `--xpu` / `--rocm` /
`--gpu auto` (see §4.1).

### `scripts/check_env.py` — verify

Read-only. Checks four layers and exits non-zero if a **required** item is
missing (optional items are `WARN` and never fail the run).

```
1. SYSTEM   OS, Python version, venv, native libs, Docker, GPU (NVIDIA / Arc / ROCm / MPS)
2. PYTHON   every package + version, plus which torch build is installed
3. ASSETS   MJCF, meshes, terrain scenes, policy weights, result CSVs
4. RUNTIME  does MuJoCo really load go2_flat.xml? do the Gym envs build?
```

```bash
python scripts/check_env.py              # full report
python scripts/check_env.py --quiet      # problems only
python scripts/check_env.py --json       # machine-readable
```

Layer 4 is the valuable part — it catches the failures that a package-version
check cannot, like a MuJoCo install that imports fine but cannot parse a model.

### `scripts/hw_profile.py` — record the machine

Stage 3 results are only comparable if you know what produced them. One
command detects CPU / RAM / every GPU vendor and `--save` writes
`hardware/profiles/<hostname>.json` for the repo.

```bash
python scripts/hw_profile.py             # detect and print
python scripts/hw_profile.py --save      # write hardware/profiles/
python scripts/hw_profile.py --list      # every recorded profile
python scripts/hw_profile.py --check     # does this host match one?
make hw-profile && make hw-check         # same via Makefile
```

`stage3-go2-training/tune_profiles.py` then picks `num_envs` / minibatch /
threads from backend + VRAM (measured profile for the RTX 3050; estimated
elsewhere until someone benchmarks).

### `scripts/gpu_check.py` — NVIDIA readiness

Answers whether **this NVIDIA host** can actually use the GPU right now
(driver match, torch CUDA matmul, EGL not silently on software). On Arc/AMD/
Apple it is not the primary tool — rely on `check_env.py` + a `--smoke` train.

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
| `GLX: Failed to create context: BadValue` | GPU driver/library mismatch — usually an NVIDIA update with no reboot since | reboot. Diagnose precisely with `make gpu`. Immediate workaround: `LIBGL_ALWAYS_SOFTWARE=1 __GLX_VENDOR_LIBRARY_NAME=mesa python …` (software rendering, slower but correct) |
| `torch.cuda.is_available()` is False with a CUDA build | same driver mismatch, or a CPU-only wheel | `make gpu` says which |
| Arc card present but `torch.xpu` missing | CPU torch installed | `./scripts/setup_env.sh --xpu` |
| AMD card, Stage 3 stays on CPU | ROCm wheel not installed / ROCm stack missing | `./scripts/setup_env.sh --rocm`; confirm `rocm-smi` works first |
| Training OOMs on a 4 GB card | default profile too large | profiles auto-pick `cuda-4gb`; or `python train.py --tune-profile cuda-4gb` |
| Training is slow on a big GPU | MuJoCo rollout is CPU-bound | expected — raise cores, not VRAM; see `tune_profiles.py` rationale |
| `MUJOCO_GL=egl` is no faster than `osmesa` | EGL silently fell back to software (`libEGL: driver (null)`) | same driver mismatch; `make gpu` detects it |
| `Image width N > framebuffer width 640` | MuJoCo's offscreen framebuffer defaults to 640×480 | set `model.vis.global_.offwidth/offheight` before constructing `mujoco.Renderer` |
| `nvidia-smi: Driver/library version mismatch` | driver updated without reboot | reboot — or ignore, Stages 1–2 are CPU-only |
| `python3 -m venv` fails | venv split into its own package | `sudo apt install python3-venv` |
| pip downloads a 2.5 GB torch | CUDA wheel resolved | install torch first from the CPU index (§4), or pass `--gpu cpu` |

Still stuck? `python scripts/check_env.py` prints exactly which layer failed
and why.
