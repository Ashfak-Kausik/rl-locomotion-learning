---
name: run
description: Launch and drive this project — the MuJoCo Go2 simulation (GUI or headless), the Stage 3 training pipeline, the experiment harness, or the figure pipeline. Use when asked to run, start, demo, or screenshot the project, or to confirm a change works in the real simulation rather than only in tests.
---

# Running rl-locomotion-learning

This repo has **no single app**. It is a research pipeline with four things
worth "running". Pick by intent:

| Intent | Command | Needs policy weights? |
|---|---|---|
| See the robot | `python stage2-go2-mujoco-inference/01_hello_go2.py` | no |
| See control working | `python stage2-go2-mujoco-inference/03_pose_go2.py` | no |
| **The flagship — robot walks** | `python stage2-go2-mujoco-inference/06_run_policy.py` | **yes** |
| Teleop with gait switching | `python stage2-go2-mujoco-inference/07_run_policy_interactive.py` | **yes** |
| Research results | `python stage2-go2-mujoco-inference/experiments/exp1_velocity_sweep.py` | **yes** |
| Figures from committed data | `python stage2-go2-mujoco-inference/experiments/make_figures.py` | no |
| Train a policy | `python stage3-go2-training/train.py --smoke` | no |
| RL fundamentals | `python stage1-rl-fundamentals/01_cartpole_ppo.py` | no |

Always activate the venv first: `source .venv/bin/activate`.

---

## Before anything else

```bash
source .venv/bin/activate
python scripts/check_env.py
```

Read the ASSETS section. If it says **`walk-these-ways checkpoints NOT FOUND`**,
that is normal — they are a large external input, gitignored on purpose, not
project output. Everything in the "no" column above still runs. They ARE
downloadable though, from the MIT-licensed Go2 fork that commits them into
git; see `docs/DEPENDENCIES.md` §6 for the verified `curl` commands.

---

## The GUI: two known failure modes

### 1. GLX context failure (this machine, as of 2026-07)

```
GLFWError: (65543) b'GLX: Failed to create context: BadValue …'
ERROR: could not create window
```

This is a **GPU driver/library mismatch** — an NVIDIA update with no reboot
since. `check_env.py` reports it as `nvidia-smi present but failing`.

**Verified workaround** (launches successfully; slower but correct):

```bash
DISPLAY=:1 LIBGL_ALWAYS_SOFTWARE=1 __GLX_VENDOR_LIBRARY_NAME=mesa \
  python stage2-go2-mujoco-inference/01_hello_go2.py
```

The real fix is a reboot. Check `DISPLAY` first — it is `:1` here, not `:0`.

### 2. No display at all (container, SSH, CI)

Use headless rendering instead — never `mujoco.viewer`:

```bash
export MUJOCO_GL=osmesa
```

---

## Seeing the result yourself

A launched window proves nothing to an agent that cannot see it. **Render
offscreen and read the PNG.**

```python
import os, mujoco, imageio, numpy as np
os.environ.setdefault("MUJOCO_GL", "osmesa")

scene = os.path.abspath("stage2-go2-mujoco-inference/scenes/go2_flat.xml")
m = mujoco.MjModel.from_xml_path(scene)          # MUST be absolute (see below)

# The offscreen framebuffer defaults to 640x480 and Renderer REFUSES anything
# larger rather than resizing. Enlarge it on the MODEL before building one.
m.vis.global_.offwidth, m.vis.global_.offheight = 900, 600

d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home"))
mujoco.mj_forward(m, d)

r = mujoco.Renderer(m, height=600, width=900)
cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
cam.distance, cam.azimuth, cam.elevation = 1.7, 130, -15
cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.18]     # follow the robot

r.update_scene(d, cam)
imageio.imwrite("/tmp/shot.png", r.render())
r.close()
```

Then `Read` the PNG. A blank or uniform frame means the launch failed.
Stack several frames with `np.hstack([...])` to show motion over time.

---

## Driving without policy weights

You can still exercise the **entire** inference pipeline: export untrained but
contract-valid networks.

```bash
python stage3-go2-training/export.py --random --out-dir /tmp/demo-policy
export GO2_POLICY_DIR=/tmp/demo-policy
python stage2-go2-mujoco-inference/experiments/harness.py   # smoke trial
```

**Expected result:** the robot stands stable but does not walk (`mean_vx ≈ 0`).
That is correct — a random net emits near-zero deltas, so joint targets ≈ the
default pose. It proves the pipeline, not the policy. Do not report this as
the robot walking.

---

## Non-negotiable gotchas

| Rule | Why |
|---|---|
| **Absolute paths** to `mujoco.MjModel.from_xml_path()` | a relative path breaks mesh resolution with a confusing "cannot open base_0.obj" |
| **Enlarge `offwidth`/`offheight`** before any `Renderer` | default 640×480; `Renderer` raises instead of resizing |
| `MUJOCO_GL=osmesa` when headless | otherwise GLFW tries to open a window and dies |
| Never `rm` inside `stage1-rl-fundamentals/tb_logs/` | those files are **tracked in git** despite matching `.gitignore` (committed before the rule existed) |
| Clean up `policies/demo-*`, `stage3-go2-training/runs/`, `*.zip` after a demo | keeps `git status` clean |

---

## Interpreting what you see

- **Robot stands, height ≈ 0.27 m, does not fall** → PD control is working.
- **Robot stands but `vx ≈ 0` with a policy loaded** → untrained/random weights.
- **Robot collapses immediately** → the training default pose override was
  likely skipped (`data.qpos[7:19] = DEFAULT_JOINT_POS` after the keyframe reset).
- **Robot twitches violently** → wrong `DECIMATION`; policy must run at 50 Hz.
- **`ret --` in the first ~6 training updates** → normal; no episode has
  finished yet.

Baseline numbers to compare against live in
`stage2-go2-mujoco-inference/experiments/results/EXPERIMENT_FINDINGS.md`.
At commanded 0.5 m/s the real policy achieves **0.227 ± 0.005 m/s**.

---

## After running

```bash
git status --short          # must be clean
```

If it is not, you left artifacts behind. Remove them — do **not** `git checkout`
whole directories, which is how tracked `tb_logs` files get destroyed.
