# Technical Design Document

**Companion to:** [SRS.md](SRS.md) (what must be true) and
[ARCHITECTURE.md](ARCHITECTURE.md) (how it fits together).
This document covers **why each design decision was made**, what was rejected,
and where the design is under strain.

Reconstructed from the implementation and its commit history, 2026-07-30.

---

## Contents

1. [Design principles](#1-design-principles)
2. [Decision record](#2-decision-record)
3. [Component design](#3-component-design)
4. [Data design](#4-data-design)
5. [Error handling](#5-error-handling)
6. [Performance design](#6-performance-design)
7. [Extension points](#7-extension-points)
8. [Known design debt](#8-known-design-debt)
9. [Design for Stage 3](#9-design-for-stage-3)

---

## 1. Design principles

These are inferred from consistent choices across the codebase, and they
explain a lot of otherwise-surprising decisions.

**P1 — Read the source, not the docs.**
Every constant in the inference pipeline was recovered from the
`walk-these-ways` training code, not from a paper or README. The observation
layout, sign conventions and scale factors are undocumented upstream; the only
authority is what the training loop actually did.

**P2 — Scripts are a curriculum, not an application.**
`01`–`08` are ordered so each introduces exactly one new concept. This
justifies duplication that would be unacceptable in production code
(see [§8.1](#81-constant-duplication)).

**P3 — Self-containment over convenience.**
Commit `2810d2b` vendored the entire Go2 model in-tree rather than depending on
an external `mujoco_menagerie` checkout. ~27 MB of meshes in git is a real
cost, accepted to buy reproducibility.

**P4 — Measure, then claim.**
Every published number traces to a committed CSV. When the traversability
metric turned out to be flawed mid-study, the fix and its rationale were
documented rather than quietly applied.

**P5 — CPU-first.**
Constraint C1 (no local GPU) shaped everything: TorchScript inference over a
training framework, headless trials over rendered ones, decimation to keep the
expensive step rare.

---

## 2. Decision record

### D1 — TorchScript for policy transport

**Context.** The policy was trained inside the `walk-these-ways` Isaac Gym
codebase, which cannot be installed here (Isaac Gym is GPU-only and its
dependency tree conflicts with MuJoCo's).

| Option | Verdict |
|---|---|
| PyTorch `state_dict` (`.pt`) | ❌ requires the training codebase's model classes |
| ONNX | ❌ extra export step and runtime; no benefit here |
| **TorchScript (`.jit`)** | ✅ self-contained graph + weights; loads with `torch.jit.load` alone |
| Reimplement the network by hand | ❌ high risk, no upside |

**Consequence.** The only PyTorch API surface in Stage 2 is `jit.load`, `eval`,
`no_grad`, `tensor`, `cat`, `numpy`. This is also the mechanism that will let
Stage 3's output slot into the existing pipeline unchanged (FR-4.6).

### D2 — Hand-written PD control instead of position actuators

**Context.** `go2.xml` declares `<motor>` actuators, so `data.ctrl[i]` is a
torque in newton-metres, not a target angle.

| Option | Verdict |
|---|---|
| Rewrite the MJCF to use `<position>` actuators | ❌ diverges from the reference model; hides the real dynamics |
| **Hand-write PD at the physics rate** | ✅ matches how the policy was trained; matches real hardware |

**Consequence.** `tau = Kp·(target − q) − Kd·qd`, evaluated every physics step
with `Kp=25, Kd=0.6` from the training config. This mirrors the actual Go2
motor controller, so the design carries forward to hardware.

### D3 — Decimation over slowing the physics

**Context.** The policy needs 50 Hz; MuJoCo runs the Go2 at 500 Hz.

| Option | Verdict |
|---|---|
| Raise `model.opt.timestep` to 0.02 s | ❌ contact solving degrades badly; robot jitters through the floor |
| **Run policy every 10th step; hold targets between** | ✅ accurate physics, correct policy rate |

**Consequence.** The `sim_step % DECIMATION == 0` pattern appears in every
runtime script. It also happens to be exactly how real robots work: a fast
motor controller under a slow planner.

### D4 — Observation history in a `deque`

**Context.** The RMA adaptation module needs the last 30 observations (2100
dims) every step.

| Option | Verdict |
|---|---|
| List + manual truncation | ❌ O(n) pops, easy to get the ordering wrong |
| Preallocated NumPy ring buffer | ~ fastest, but index bookkeeping obscures the concept |
| **`deque(maxlen=30)`** | ✅ O(1), self-truncating, reads exactly like the concept |

**Consequence.** `np.concatenate(obs_history)` is O(2100) per step — trivial
next to the network forward pass. Correct trade for this workload.

### D5 — Metrics dict returned, CSV written by the caller

**Context.** Three experiments need the same trial machinery with different
sweeps.

`run_trial()` returns a plain dict and performs no I/O. Each experiment script
owns its own CSV schema and summary table.

**Consequence.** `harness.py` is importable and unit-testable in principle,
and its `__main__` block is a smoke test. Experiments stay thin — `exp2` is
127 lines, almost all of it presentation. This is the cleanest module in the
repo and the right template for new work.

### D6 — Seeded initial-condition randomisation

**Context.** Deterministic trials from an identical start would produce
zero variance, making mean ± std meaningless.

```python
rng = np.random.default_rng(seed)
data.qpos[7:19] += rng.uniform(-0.02, 0.02, size=12)   # ±1.1°
data.qpos[2]    += rng.uniform(-0.01, 0.01)            # ±1 cm
yaw              = rng.uniform(-0.05, 0.05)            # ±2.9°
data.qvel[6:18] += rng.uniform(-0.05, 0.05, size=12)
```

**Rationale.** Magnitudes represent realistic deployment variability — no two
robot starts are identical — while being small enough not to dominate the
effect under study. `default_rng(seed)` guarantees trial *k* is reproducible
forever.

**Consequence.** Reported std values are meaningful. Experiment 1's finding
that variance *grows with commanded velocity* (F1.5) is only detectable
because of this design.

### D7 — Terrain as generated-but-committed XML

| Option | Verdict |
|---|---|
| Generate at trial time | ❌ a generator edit silently changes past results |
| Hand-write 11 XMLs | ❌ unmaintainable, error-prone |
| **Generate once, commit the output** | ✅ exact geometry preserved; regeneration verifiable |

**Consequence.** `git diff` after re-running the generator must be empty —
which it is (verified). Cheap, strong reproducibility guarantee.

### D8 — Fall thresholds at 0.15 m and 60°

Nominal standing height is ~0.30 m, so **0.15 m is half of it** — unambiguously
collapsed, not merely crouching. **60°** is past the tilt from which a
quadruped can self-recover. Both are checked every physics step, so
`fall_time_s` has 2 ms resolution.

**Consequence.** Falls abort the trial immediately (`break`), and all metrics
are recorded as `null` rather than averaged from a partial run.

### D9 — Traversability requires progress, not just survival ⭐

**The most important design decision in the repo, and it was a correction.**

Original: `traversable = survival ≥ 80%`. Experiment 3 then showed 100%
survival on 16 cm stairs — which is absurd. The `distance_traveled` column
explained it: a deterministic 0.47 m across all seeds, against a 2.5 m run-up.
The robot was walking up to the first step and stably standing there forever.

```python
PROGRESS_THRESHOLD_M = 3.5     # 2.5 m run-up + >=1 m genuine terrain progress
traversable = (surv_rate >= 0.8) and (mean_dist > PROGRESS_THRESHOLD_M)
```

**Consequence.** A new failure mode was named — **safe stall**: survives, does
not progress. Survival-only metrics misclassify it as success. This is a
methodological result about how locomotion policies should be evaluated, and it
generalises beyond this project.

### D10 — Environment-variable path resolution (added in this branch)

**Context.** Ten constants across eight files hardcoded `/home/user/projects/…`.

| Option | Verdict |
|---|---|
| Config file (YAML/TOML) | ❌ adds a dependency and a parse step for two strings |
| CLI arguments everywhere | ❌ invasive; breaks the "just run it" ergonomics |
| **Env vars with in-repo defaults** | ✅ zero-config default, Docker-native, one file to change |

**Consequence.** `paths.py` is the single source of truth. `GO2_POLICY_DIR`
maps directly onto a Docker volume mount. Nothing else in the pipeline changed.

---

## 3. Component design

### 3.1 `paths.py` — path resolution

```
Responsibility : resolve external artefact locations; nothing else
Dependencies   : os, sys, pathlib (torch imported lazily inside load_policy)
Interface      : MODEL_PATH, POLICY_DIR, SCENES_DIR,
                 policy_paths(), policy_available(),
                 require_policy(), load_policy()
```

`torch` is imported *inside* `load_policy()` so scripts `01`–`04` — which need
MuJoCo but not PyTorch — can import this module without paying for torch.

`require_policy()` raises `FileNotFoundError` with a multi-line message naming
the missing files, the env var to set, and what still works without them. An
error message is documentation delivered at the moment of need.

### 3.2 `harness.py` — the trial engine

```
Responsibility : execute one headless trial, return metrics
Dependencies   : mujoco, torch, numpy, paths
Interface      : run_trial(scene_path, lin_vel_x, lin_vel_y, ang_vel_yaw,
                           gait, settle_s, measure_s,
                           body_net, adapt_net, seed) -> dict
```

Three deliberate choices:

1. **Networks injected, not loaded.** `body_net`/`adapt_net` are parameters, so
   a 75-trial sweep does one `jit.load` instead of 75. Defaults to loading if
   omitted, so the smoke test stays a one-liner.
2. **No I/O.** Returns a dict. Callers own persistence.
3. **No rendering, no sleeping.** The two things that make the interactive
   scripts real-time are simply absent, which is the entire speed-up.

### 3.3 Experiment scripts — sweep + present

All three follow one shape:

```python
body_net, adapt_net = load once
for condition in CONDITIONS:
    for seed in range(N_TRIALS):
        rows.append(run_trial(..., seed=seed))
write_csv(rows)
print_summary(rows)          # mean ± std over SURVIVING trials only
```

Aggregating over survivors only is important: a fallen trial has `null`
metrics, and mixing them in would silently bias results toward whatever
partial behaviour preceded the fall.

### 3.4 `generate_terrain_scenes.py` — parametric MJCF

Shared `HEADER`/`FOOTER` templates plus two builders. Geometry is computed, not
hand-tuned:

```python
a  = np.radians(angle_deg)
cx = x_start + (ramp_len / 2.0) * np.cos(a)
cz = (ramp_len / 2.0) * np.sin(a) - (ramp_thick / 2.0) * np.cos(a)
```

Every scene shares a 2.5 m flat run-up so the policy reaches steady state
before meeting the terrain — without it, the measurement would be dominated by
start-up transients. The run-up length is also what makes the 3.5 m progress
threshold (D9) interpretable.

### 3.5 `make_figures.py` — CSV → PNG

```
Dependencies : csv, numpy, matplotlib(Agg)   ← note: NOT mujoco, NOT torch
```

That dependency list is the design. Figures regenerate from committed data with
no simulator and no policy weights, so a reviewer can reproduce every plot in
seconds on any machine.

### 3.6 `check_env.py` — verification

Four layers, deliberately ordered cheapest-to-most-informative: system →
Python packages → assets → **runtime**. Layer 4 is the one that matters: it
actually loads a MuJoCo model and builds Gym environments, catching installs
that import cleanly but cannot do the job.

Capability-based rather than package-name-based: it checks for `Python.h` and a
working `cc`, not for `python3-dev` and `build-essential`, so it is correct on
any distro and inside slim container images.

---

## 4. Data design

### 4.1 The observation contract

The 70-dim layout is an **immutable contract** with the pretrained policy
(constraint C2). It cannot be improved, reordered or extended without
retraining. Full field map in
[ARCHITECTURE.md §7](ARCHITECTURE.md#7-the-70-dimensional-observation-vector).

Defended by `assert obs.shape == (70,)`. A shape assert catches gross errors;
it cannot catch a *reordering*, which is why `harness.py` — not
`04_build_obs_vector.py` — must be the reference implementation.

### 4.2 State slicing

`qpos[0:3] | qpos[3:7] | qpos[7:19]` and `qvel[0:3] | qvel[3:6] | qvel[6:18]`.
The offset difference (7 vs 6) comes from the free joint's quaternion needing
4 numbers in position space but 3 in velocity space. These slices appear
verbatim throughout the codebase — a named-accessor helper would be a
reasonable improvement, at the cost of one more indirection between the reader
and MuJoCo's actual API.

### 4.3 CSV as the results format

Chosen over a database or Parquet because: human-readable in review, diffable
in git, `csv` is stdlib, and 75 rows do not need indexing. `null` for fallen
trials preserves the distinction between "zero" and "not measured".

---

## 5. Error handling

The codebase uses **fail-fast with actionable messages**, not defensive
recovery. For research code this is correct: silently degrading is far worse
than stopping.

| Failure | Handling | Rationale |
|---|---|---|
| Wrong observation shape | `assert` | catches layout bugs at the source |
| Missing policy weights | `require_policy()` with guidance | most likely newcomer failure |
| Missing scene file | `08` lists available scenes, exits 1 | typo recovery |
| Missing `home` keyframe | `if key_id >= 0` guard | tolerates scenes without one |
| Robot falls | `break`, record time, `null` metrics | a fall is data, not an error |
| Fell before the measure window | `start_pos is None` guard | avoids a crash on a degenerate trial |
| Missing dependency | `check_env.py` names it and how to fix | pre-flight, not mid-run |

**Gap:** nothing validates that a *loaded* policy has the expected input
dimensions. Feeding a 2102-dim tensor to a network expecting something else
produces a torch error rather than "this checkpoint is not the one this
pipeline expects". A shape probe in `load_policy()` would close this cheaply.

---

## 6. Performance design

| Concern | Design | Effect |
|---|---|---|
| Inference cost | run at 50 Hz, not 500 Hz | 10× fewer forward passes |
| Gradient overhead | `torch.no_grad()` | ~2× faster, less memory |
| Repeated model loads | inject nets into `run_trial` | 75 loads → 1 |
| Rendering | absent in the harness | the single biggest speed-up |
| Real-time pacing | absent in the harness | trials run as fast as the CPU allows |
| History flattening | `np.concatenate` per step | O(2100), negligible |
| Torch wheel size | CPU-only index | 2.5 GB → 190 MB |
| Docker layer caching | `requirements.txt` copied before code | code edits never trigger reinstall |

Measured budget: 50 Hz gives 20 ms per policy step; inference is reported at
< 2 ms, leaving ~90% headroom. Adding a vision encoder in Stage 4 will consume
much of that — worth remembering now.

---

## 7. Extension points

Designed-in seams for the work ahead.

### 7.1 New terrain

Add a builder to `generate_terrain_scenes.py`, run it, commit the XML, and add
the condition to `exp3_terrain.py`. No harness change.

### 7.2 New gait

Add an entry to `GAIT_PRESETS` — a `(phase, offset, bound)` triple. Because
gait conditioning is continuous, intermediate values are legal and unexplored:
`(0.25, 0.25, 0.0)` is a perfectly valid experiment nobody has run.

### 7.3 New metric

Log the quantity inside `run_trial`'s measurement loop, add it to the returned
dict and to each script's `CSV_FIELDS`. Additive; existing CSVs stay readable.

### 7.4 New policy

Point `GO2_POLICY_DIR` at another checkpoint directory. Any policy honouring
the 70/2100/2102/12 contract works unmodified — **this is how Stage 3's output
will be evaluated.**

### 7.5 New scene source

Set `GO2_MODEL_PATH`. Useful for testing against a different robot MJCF.

---

## 8. Known design debt

### 8.1 Constant duplication

`DEFAULT_JOINT_POS`, `OBS_SCALES`, `build_obs`, `quat_rotate_inverse` and the
PD constants exist in four copies (`06`, `07`, `08`, `harness.py`), currently
**semantically identical** (verified line-by-line).

*Why it exists:* principle P2 — the numbered scripts are teaching artefacts
meant to be read standalone.

*Risk:* silent divergence. A fix applied to `harness.py` will not reach `07`.

*Recommended resolution:* extract `go2_common.py` for `07`, `08` and
`harness.py` only. Freeze `01`–`06` as curriculum. Full reasoning in
[REVERSE-ENGINEERING.md §5](REVERSE-ENGINEERING.md#5-the-duplication-question).

### 8.2 No test suite

The highest-priority gap. Cheap, high-value first tests, none of which need
policy weights:

```python
def test_obs_shape():            # build_obs returns exactly (70,)
def test_model_dimensions():     # go2_flat.xml gives nq=19, nv=18, nu=12
def test_scene_regeneration():   # regenerated XMLs match committed ones
def test_hip_indices():          # actuators 0,3,6,9 are the hip joints
def test_quat_rotate_inverse():  # identity quaternion is a no-op
def test_gait_presets():         # each preset yields distinct clock signals
```

### 8.3 Hardcoded sweep parameters

Velocities, gaits, seeds and durations are module constants. Any variation
requires editing source, which is not diff-friendly for a research log.
`argparse` would fix this without changing defaults.

### 8.4 No run provenance

CSVs record results but not the git SHA, package versions or wall-clock time
that produced them. Six months from now, reproducing an exact number will be
guesswork. Two extra columns would solve it.

### 8.5 `04_build_obs_vector.py` is actively misleading

It contains a superseded, incorrect layout with no warning in the file itself.
A header banner costs nothing and prevents a real class of bug.

---

## 9. Design for Stage 3

The next major piece of work. Designing it against the existing seams keeps the
whole Stage 2 measurement apparatus usable on day one.

### 9.1 Constraints

- **Bursty GPU.** Free-tier sessions are time-limited and can be killed.
  Checkpoint frequently and support resume-from-checkpoint (FR-4.5).
- **Contract compatibility.** Emit the same 70-dim observation and 12-dim
  action (FR-4.6), so `harness.py` evaluates the new policy unchanged.
- **Terrain curriculum.** Experiment 3 found 10° slopes traversable and 5 cm
  steps not. That is a ready-made curriculum schedule, derived from measurement
  rather than guesswork.

### 9.2 Proposed shape

```
stage3-go2-training/
├── env/
│   ├── go2_env.py           MJX/mujoco_playground env, 70-dim obs contract
│   ├── rewards.py           velocity tracking, gait regularity, energy, survival
│   ├── curriculum.py        terrain schedule seeded by Exp 3 thresholds
│   └── domain_rand.py       friction, mass, motor strength, latency
├── train.py                 PPO loop, checkpoint every N steps
├── export.py                trained policy -> TorchScript in the SAME contract
└── configs/
```

### 9.3 The critical interface

```
  Stage 3 export.py                     Stage 2 (unchanged)
  ┌──────────────────────┐              ┌─────────────────────────┐
  │ trained policy       │              │ paths.py                │
  │   ↓ TorchScript      │─────────────►│   GO2_POLICY_DIR=...    │
  │ body_latest.jit      │              │ harness.py::run_trial   │
  │ adaptation_...jit    │              │ exp1/exp2/exp3          │
  └──────────────────────┘              └─────────────────────────┘
       must honour:  obs 70 · history 30 · latent 2 · action 12
```

Honour that contract and the new policy is immediately measurable against the
existing baselines with the existing harness, on the existing terrain, using
the existing figures. That comparability is the entire value of Stage 2's
measurement work — do not break it for convenience.

---

## Related

- [ARCHITECTURE.md](ARCHITECTURE.md) — structural view with diagrams
- [SRS.md](SRS.md) — the requirements this design satisfies
- [REVERSE-ENGINEERING.md](REVERSE-ENGINEERING.md) — findings and first tasks
- [FEATURES.md](FEATURES.md) — feature matrix
