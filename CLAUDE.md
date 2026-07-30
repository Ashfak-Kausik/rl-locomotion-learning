# CLAUDE.md

Project context for AI coding assistants. Read this before editing anything.
Full guidance: [`docs/AGENTIC-WORKFLOW.md`](docs/AGENTIC-WORKFLOW.md).

---

## What this project is

A research repository that deploys a quadruped locomotion policy **trained
elsewhere** (`walk-these-ways`, NVIDIA Isaac Gym, GPU cluster) into **MuJoCo on
CPU**, and measures rigorously how much performance is lost in that transfer.

Two delivered stages plus a research layer:

- `stage1-rl-fundamentals/` — PPO on CartPole / LunarLander / Pendulum (complete)
- `stage2-go2-mujoco-inference/` — the Go2 inference pipeline (complete)
- `stage2-go2-mujoco-inference/experiments/` — 3 studies, 75 trials (complete)
- Stage 3 (custom training) is declared "in progress" but **no code exists yet**

---

## Hard invariants — do not change these without explicit instruction

### 1. The observation contract is external and immutable

The policy is a pretrained binary. Its input format cannot be renegotiated.

```
70-dim observation, in this exact order:
  [0:3]   projected gravity (body frame)
  [3:18]  15 scaled commands
  [18:30] joint positions relative to DEFAULT_JOINT_POS
  [30:42] joint velocities × 0.05
  [42:54] last action  (t−1)
  [54:66] prev action  (t−2)
  [66:70] 4 per-foot clock signals
History: 30 frames → 2100 dims.  Body input: 2100 + 2 latent = 2102.  Output: 12.
```

**Never** reorder fields, change `OBS_SCALES`, or alter the history length.
A wrong value fails **silently** — the robot just walks worse. There is no test
that catches it.

**Reference implementation: `experiments/harness.py`.**

### 2. `04_build_obs_vector.py` is SUPERSEDED and known-wrong

It has a different field order *and* inverted hip signs. It is kept as a
learning artefact from before the layout was recovered correctly. **Never copy
from it.** Copy from `harness.py`.

### 3. Control constants come from the training config

```python
DEFAULT_JOINT_POS = [ 0.1, 0.8, -1.5,   # FL   hip sign POSITIVE  on the left
                     -0.1, 0.8, -1.5,   # FR   hip sign NEGATIVE on the right
                      0.1, 1.0, -1.5,   # RL
                     -0.1, 1.0, -1.5]   # RR
ACTION_SCALE = 0.25          # ×0.5 additionally on hips (indices 0, 3, 6, 9)
KP, KD = 25.0, 0.6           # PD gains, applied every physics step
DECIMATION = 10              # 500 Hz physics ÷ 10 = 50 Hz policy
HISTORY_LEN, OBS_DIM = 30, 70
```

The alternating hip signs look like a typo. They are not — verified against
`scenes/go2_model/go2.xml` lines 189–200, where actuator order is
`FL, FR, RL, RR`.

`DECIMATION = 10` is not a tuning knob. The policy was trained at 50 Hz.

### 4. Scripts `01`–`06` are a curriculum, not an application

The duplication between them (`build_obs`, `quat_rotate_inverse`,
`DEFAULT_JOINT_POS`, PD constants) is **intentional** — each script is meant to
be read standalone, top to bottom. Do not extract shared modules from `01`–`06`.

`07`, `08` and `harness.py` are tools, not lessons; consolidating those three is
a reasonable proposal. See
[`docs/REVERSE-ENGINEERING.md` §5](docs/REVERSE-ENGINEERING.md#5-the-duplication-question).

### 5. Two poses exist, and both are correct

The MJCF `home` keyframe (`0, 0.9, −1.8`) is **not** the training default pose.
Every runtime script resets to the keyframe and then overrides:

```python
mujoco.mj_resetDataKeyframe(model, data, key_id)
data.qpos[7:19] = DEFAULT_JOINT_POS
data.qpos[2] = 0.30
mujoco.mj_forward(model, data)
```

Removing the override silently offsets all 12 joint observations.

### 6. Always pass absolute paths to MuJoCo

`mujoco.MjModel.from_xml_path()` fails to resolve meshes for these scenes when
given a relative path. `paths.py` always produces absolute paths; keep it that way.

---

## Layout you need to know

```
stage2-go2-mujoco-inference/
├── paths.py          ← ALL external paths resolve here (env-var overridable)
├── 01-08_*.py        ← numbered curriculum; 06 is the canonical walking demo
└── experiments/
    ├── harness.py    ← reference implementation; run_trial() is the core
    ├── exp1/2/3      ← thin sweeps over run_trial
    └── make_figures.py ← CSV → PNG; needs NO policy weights
scripts/check_env.py  ← 4-layer environment verifier
docs/                 ← architecture, SRS, TDD, features, findings
```

Environment variables (`paths.py`):
`GO2_POLICY_DIR`, `GO2_SCENE`, `GO2_MODEL_PATH`, `GO2_SCENES_DIR`.

---

## Missing policy weights are normal

`policies/walk-these-ways-go2/{body_latest.jit,adaptation_module_latest.jit}`
are **not** in the repo. They are a large external input, not project output.

**Do not invent a download URL for them.** There is no canonical public one.
See [`docs/DEPENDENCIES.md` §6](docs/DEPENDENCIES.md#6-the-one-dependency-we-cannot-install).

Plenty works without them: all of Stage 1, Stage 2 scripts `01`–`04`, scene
generation, terrain inspection, and `make_figures.py` (which regenerates all
four data figures from committed CSVs).

---

## Verification — cheap, fast, no weights required

Run these and paste the output rather than asserting success.

```bash
source .venv/bin/activate
python scripts/check_env.py                                    # 4-layer report
python -m py_compile stage2-go2-mujoco-inference/*.py           # syntax
python stage2-go2-mujoco-inference/paths.py                     # path resolution
python stage2-go2-mujoco-inference/experiments/make_figures.py  # full data path
python stage2-go2-mujoco-inference/experiments/generate_terrain_scenes.py \
  && git diff --stat stage2-go2-mujoco-inference/scenes/        # must be EMPTY
```

Container equivalents in [`docs/DOCKER.md`](docs/DOCKER.md).

---

## Conventions

- **Commits carry no agentic footprint.** Never add `Co-Authored-By: Claude`,
  "Generated with …", emoji trailers, or any other marker indicating an AI
  wrote the change. Commit messages describe the change and its rationale,
  nothing else. This applies to commit messages, PR bodies, and code comments.
- **Style:** 4-space indent, snake_case, section banners (`# ==== NAME ====`),
  module docstrings explaining *why* the script exists. Match the surrounding file.
- **Comments** explain physics and provenance, not syntax. Keep that habit.
- **Experiments are append-only.** To run a new sweep, write `exp4_*.py` — do
  **not** edit constants in `exp1/2/3`, or past results stop being reproducible
  from committed source.
- **`null` metrics for fallen trials are deliberate.** Do not coerce them to 0.
- **Findings are numbered** (F1.1, F3.4) in
  `experiments/results/EXPERIMENT_FINDINGS.md` so later work can cite them.
  Append new sections; never overwrite.
- **No test suite exists.** This is the highest-priority gap. Any refactor of
  `harness.py` should add tests first.

---

## Known issues (details in `docs/REVERSE-ENGINEERING.md`)

| | Issue |
|---|---|
| R4 | `04_build_obs_vector.py` superseded, no warning in the file |
| R5 | hip-index comment in `06_run_policy.py` names the legs in the wrong order (indices are right) |
| R8 | `06_run_policy.py:240` says "30 seconds", code runs 60 |
| R12 | root `README.md` truncated at "Repository Structure" |
| — | README headline claims ~0.28 m/s; Exp 1 measured 0.227 ± 0.005 at the same command |

---

## Good first tasks

1. **A test suite** (`tests/`, pytest, no policy weights needed) — highest value.
2. **CI** running `check_env.py` + those tests.
3. **Gait × velocity grid** — Exp 2 found pace beats trot at a *single* speed;
   the findings log flags that limitation itself. ~90 trials, existing code.
4. Banner `04_build_obs_vector.py` as superseded; fix R5 and R8.

---

## Documentation map

| File | Read when |
|---|---|
| [`docs/README.md`](docs/README.md) | starting out — has the reading order |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | you need the system view + diagrams |
| [`docs/TECH-STACK-PRIMER.md`](docs/TECH-STACK-PRIMER.md) | MuJoCo/PyTorch/Gymnasium are new to you |
| [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md) | setting up or something will not install |
| [`docs/DOCKER.md`](docs/DOCKER.md) | running in a container |
| [`docs/SRS.md`](docs/SRS.md) | you need the requirements |
| [`docs/TDD.md`](docs/TDD.md) | you need to know *why* it is built this way |
| [`docs/FEATURES.md`](docs/FEATURES.md) | you need the implemented/planned split |
| [`docs/REVERSE-ENGINEERING.md`](docs/REVERSE-ENGINEERING.md) | you need known issues and first tasks |
| [`docs/AGENTIC-WORKFLOW.md`](docs/AGENTIC-WORKFLOW.md) | working with AI tools on this repo |
| `reinforcement-learning-theories/` | you need the RL theory (8 chapters) |
