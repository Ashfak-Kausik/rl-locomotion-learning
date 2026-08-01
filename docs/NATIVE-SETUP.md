# Native setup — running without Docker

A record of what is installed **directly on the host** to run this project,
and proof that each piece works. Docker is not used here; the container path
is still documented in [`DOCKER.md`](DOCKER.md) and remains valid, it is just
not the route this machine takes.

If you are setting up a *new* machine, do not follow this file by hand — run
the standalone bootstrap, which derives the same list and skips whatever is
already present:

```bash
./scripts/setup_env.sh --gpu auto   # OS packages + .venv + matching torch
source .venv/bin/activate
python scripts/check_env.py         # 4-layer verification
make hw-profile                     # record this machine in hardware/profiles/
```

GPU vendor matrix (NVIDIA / Intel Arc / AMD ROCm / Apple) and what to do on
each: [`DEPENDENCIES.md` §4.1](DEPENDENCIES.md#41-gpu-flavours--nvidia-intel-arc-amd-apple).

This file exists for the other question: *what did that actually put on my
machine, and can I undo it?*

---

## 1. Reference machine

| | |
|---|---|
| OS | Pop!\_OS 24.04 LTS (Ubuntu 24.04 base, `apt`) |
| Kernel | 7.0.11-76070011-generic |
| Python | 3.12.3 (system), same inside `.venv` |
| GPU | NVIDIA GeForce RTX 3050, 8 GB, sm_86 |
| NVIDIA driver | 580.173.02 (`nvidia-driver-580-open`, DKMS) |

---

## 2. System packages (apt)

All 13 are now present. Sizes are small; the whole set is a few tens of MB.

| Package | Version | Needed for |
|---|---|---|
| `build-essential` | 12.10ubuntu1 | compiles Box2D (LunarLander-v3) |
| `python3-dev` | 3.12.3-0ubuntu2.1 | C headers for the same |
| `python3-venv` | 3.12.3-0ubuntu2.1 | creating `.venv` |
| `python3-pip` | 24.0+dfsg-1ubuntu1.3 | installing into it |
| `pkg-config` | 1.8.1-2build1 | build-time library discovery |
| `swig` | 4.2.0-2ubuntu1 | Box2D bindings, if no wheel exists |
| `libgl1` | 1.7.0-1build1 | OpenGL loader |
| `libglfw3` | 3.3.10-1build1 | **the GUI viewer window** (`MUJOCO_GL=glfw`) |
| `libegl1` | 1.7.0-1build1 | GPU offscreen rendering (`MUJOCO_GL=egl`) |
| `libosmesa6` | 25.1.7-1ubuntu2~24.04.2 | CPU offscreen rendering (`MUJOCO_GL=osmesa`) |
| `ffmpeg` | 7:6.1.1-3ubuntu5 | encoding rendered frames to video |
| `patchelf` | 0.18.0-1.1build1 | occasionally needed by MuJoCo native deps |
| `git` | 1:2.43.0-1ubuntu7.3 | version control |

Four of these were installed for this project; the rest came with the OS or a
prior toolchain:

```bash
sudo apt-get install -y libglfw3 swig patchelf pkg-config
```

`libglfw3` is the one that changed behaviour — without it there is no viewer
window, and every GUI script (`01`, `03`, `04`, `06`, `07`, `08`) fails.

To remove just those four: `sudo apt-get remove libglfw3 swig patchelf pkg-config`.
Nothing else on the system was modified.

---

## 3. Python packages

Everything lives in `.venv` — **nothing is installed system-wide**, so removal
is `rm -rf .venv`. Current size **5.1 GB**, almost entirely the CUDA torch
wheel.

| Package | Version |
|---|---|
| numpy | 2.5.1 |
| mujoco | 3.11.0 |
| torch | **2.13.0+cu130** |
| matplotlib | 3.11.1 |
| imageio | 2.37.4 |
| gymnasium | 1.3.0 |
| Box2D | 2.3.10 |
| stable-baselines3 | 2.9.0 |
| tensorboard | 2.21.0 |
| tqdm | 4.70.0 |
| rich | 15.0.0 |
| pytest | 9.1.1 |

### The one deviation from `requirements.txt`

`requirements.txt` targets the **CPU** torch wheel (~200 MB), which is the
right default for a repo that must run anywhere. This machine has a usable
GPU, so it runs the CUDA build instead:

```bash
pip install 'torch==2.13.0+cu130' --index-url https://download.pytorch.org/whl/cu130
```

That is a superset — it satisfies `torch>=2.4,<3` and every CPU code path
still works — so `requirements.txt` is deliberately left alone. `cu130` rather
than `cu128` because cu128 tops out at torch 2.11.

To go back to CPU-only: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

---

## 4. What is deliberately NOT installed

| | Why |
|---|---|
| `nvidia-container-toolkit` | only needed for the Docker `gpu` service; Docker is unused here |
| `mujoco-mjx`, `jax[cuda12]` | the MJX port is designed but not written — see Stage 3 README |
| walk-these-ways policy weights (in **this repo's git**) | not committed here on purpose — `policies/` is gitignored, they are a large external input. **They ARE downloadable** — see below and [`DEPENDENCIES.md` §6](DEPENDENCIES.md#6-the-policy-weights-are-not-committed-here-but-are-downloadable) |

The real `walk-these-ways-go2` weights were downloaded and verified on this
machine on 2026-07-31. The Go2 fork commits its pretrained checkpoint into
git, MIT licensed: <https://github.com/Teddy-Liao/walk-these-ways-go2>.

```bash
mkdir -p policies/walk-these-ways-go2
BASE="https://raw.githubusercontent.com/Teddy-Liao/walk-these-ways-go2/main/runs/gait-conditioned-agility/pretrain-go2/train/142238.667503/checkpoints"
curl -sL --fail "$BASE/body_latest.jit"              -o policies/walk-these-ways-go2/body_latest.jit
curl -sL --fail "$BASE/adaptation_module_latest.jit" -o policies/walk-these-ways-go2/adaptation_module_latest.jit
python stage2-go2-mujoco-inference/paths.py           # confirms both OK
```

Without them, scripts `05`–`08` and experiments 1–3 are the only things
blocked; `make_figures.py` still regenerates all four data figures from
committed CSVs. A Stage 3 export can stand in too, if you want an untrained
control condition instead of the real weights:

```bash
python stage3-go2-training/train.py --smoke
python stage3-go2-training/export.py --checkpoint stage3-go2-training/runs/smoke/checkpoint_final.pt
export GO2_POLICY_DIR=$PWD/policies/stage3-smoke     # now 05-08 run, but stand still (untrained)
```

---

## 5. Verified working

Each of these was run on this machine, in this configuration:

| Command | Result |
|---|---|
| `make test` | 92 passed (94 including `-m slow`) |
| `python scripts/check_env.py` | READY, 1 optional item missing (the weights) |
| `make gpu` | all 7 checks OK |
| `make gpu-bench` | matmul 7.1x, EGL render 47x vs CPU |
| `python stage3-go2-training/train.py --smoke` | both RMA phases, contract PASS, on CUDA |
| `export.py` → `evaluate.py` | full round trip through Stage 2's harness |
| `experiments/make_figures.py` | all 4 figures regenerated |
| `generate_terrain_scenes.py` | empty git diff — reproducible |
| Stage 1 CartPole / LunarLander / Pendulum | all three train via SB3 |
| `08_view_scene.py` (glfw, `DISPLAY=:1`) | viewer window opens and renders |

### Rendering backends

All three work. Pick by task:

| `MUJOCO_GL` | Speed (1280x720) | Use for |
|---|---|---|
| `glfw` | interactive | watching the robot in a window |
| `egl` | **309 fps** | headless capture, experiment sweeps |
| `osmesa` | 6.6 fps | fallback when the GPU is unavailable |

**`egl` can lie.** When it cannot load the driver it returns valid frames at
software speed and raises nothing — the 6.6-vs-309 gap is otherwise invisible.
`scripts/gpu_check.py` detects this by looking for `driver (null)` on stderr.
Trust `make gpu`, not "it rendered".

---

## 6. Gotchas seen on this machine

**Driver upgraded while running.** `apt` replaced the NVIDIA userspace
libraries at 580.173.02 while the kernel module in memory was still
580.159.03. Symptoms: `torch.cuda.is_available()` False with
`Error 804: forward compatibility was attempted on non supported HW`, and EGL
silently falling back to software. The module cannot reload live because the
display server holds it. **Fix: reboot.** Nothing to install. It recurs only
after another driver or kernel upgrade, and `unattended-upgrades` is not
installed here, so it will not happen unprompted.

**`GLX: Failed to create context`** has the same root cause. The
`LIBGL_ALWAYS_SOFTWARE=1` workaround in `CLAUDE.md` gets you running before
the reboot; it is not needed after.

**The GUI viewer exits 139 (SIGSEGV) even on a fully successful run.** Every
script using `mujoco.viewer.launch_passive` — `01`, `03`, `04`, `06`, `07`,
`08` — segfaults during interpreter shutdown, *after* the simulation loop and
the context manager have both completed. Confirmed with a minimal repro:

```
SIM LOOP OK, leaving context manager
CONTEXT EXITED CLEANLY
exit=139
```

Results are unaffected — the crash is in GL/GLFW teardown, once all Python
work is done and output is flushed. But **the exit code is a lie**, so these
scripts cannot be used in `make`, CI, or any `cmd && next` chain without
guarding. Ending the script with `os._exit(0)` skips the teardown and exits 0;
verified. The curriculum scripts are deliberately left unpatched, since they
are meant to be read as lessons rather than wired into automation.

Offscreen rendering (`egl`, `osmesa`) does not use the viewer and is
unaffected — prefer it for anything scripted.

**`stage1-rl-fundamentals/tb_logs/` is tracked in git** despite matching
`.gitignore`. Never `rm` it.

---

## Related

- [`DEPENDENCIES.md`](DEPENDENCIES.md) — the full checklist and why each item exists
- [`DEPENDENCIES.md` §4.1](DEPENDENCIES.md#41-gpu-flavours--nvidia-intel-arc-amd-apple) — NVIDIA / Arc / ROCm / MPS install paths
- [`DOCKER.md`](DOCKER.md) — the container route, if you change your mind
- `hardware/profiles/` — committed machine records (`make hw-profile`)
- `stage3-go2-training/tune_profiles.py` — per-backend PPO batch sizes
- [`../stage3-go2-training/README.md`](../stage3-go2-training/README.md) — GPU tuning and measured speedups
