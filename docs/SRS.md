# Software Requirements Specification

**Project:** rl-locomotion-learning — Quadruped Locomotion via Reinforcement Learning
**Version:** 1.0 (reverse-engineered from the implementation, 2026-07-30)
**Status:** Stages 1–3 delivered (Stage 3 pipeline complete, awaiting GPU compute)

> This SRS was written *after* the fact, by reading the code. Requirements
> marked **✅** are implemented and verified. Requirements marked **📋** are
> derived from the project roadmap and are not yet built. That split is the
> whole point of the document — it makes the gap between intent and reality
> explicit rather than aspirational.

---

## 1. Introduction

### 1.1 Purpose

Specify the functional and non-functional requirements of a research platform
for deploying, evaluating and (eventually) training reinforcement-learning
locomotion controllers for the Unitree Go2 quadruped.

### 1.2 Scope

The system covers four capabilities:

1. **Learn** — reference PPO implementations on standard benchmarks, used to
   establish algorithmic understanding and diagnostic literacy.
2. **Deploy** — run a policy trained in one simulator (Isaac Gym) inside a
   different one (MuJoCo) on commodity CPU hardware.
3. **Measure** — quantify, reproducibly, how much locomotion performance is
   lost in that transfer.
4. **Extend** — provide the foundation for training custom policies,
   vision-conditioned control, and hardware deployment.

Out of scope: hardware safety certification, real-time OS guarantees,
multi-robot coordination, commercial deployment.

### 1.3 Definitions

| Term | Meaning |
|---|---|
| **Policy** | Neural network mapping observations to actions |
| **Observation** | 70-dim vector the policy consumes each control step |
| **Action** | 12 joint-position *deltas* from the default pose |
| **Decimation** | Physics steps executed per policy step (10) |
| **Sim-to-sim** | Transferring a policy between simulators |
| **Sim-to-real** | Transferring a policy from simulation to hardware |
| **RMA** | Rapid Motor Adaptation — infers environment latents from proprioceptive history |
| **Gait conditioning** | Selecting trot/pace/bound via continuous command inputs |
| **Trial** | One seeded simulation run producing one metrics record |
| **Traversable** | Survives **and** makes > 3.5 m forward progress |
| **Safe stall** | Survives without falling but makes no progress |

### 1.4 Stakeholders

| Stakeholder | Primary need |
|---|---|
| Researcher / author | reproducible experiments, publishable results |
| New contributor | a runnable environment and a comprehensible codebase |
| Reviewer | verifiable claims traceable to raw data |
| Downstream user | a policy that can be deployed to hardware |

---

## 2. Overall description

### 2.1 Product perspective

A self-contained research repository. It consumes a pretrained policy as an
external input and produces measurements, figures and findings as output.

```
  EXTERNAL INPUT              SYSTEM                     OUTPUT
  ─────────────────────────────────────────────────────────────────────
  walk-these-ways .jit   →   inference pipeline    →   locomotion behaviour
  Go2 MJCF (vendored)    →   experiment harness    →   CSV metrics
  user commands          →   figure pipeline       →   PNG figures
                                                   →   documented findings
```

### 2.2 User classes

| Class | Expertise | Typical entry point |
|---|---|---|
| RL newcomer | can program, new to RL | `docs/TECH-STACK-PRIMER.md`, Stage 1 |
| Robotics engineer | knows control, new to RL | Stage 2, `ARCHITECTURE.md` |
| RL researcher | knows RL, new to this repo | `harness.py`, `EXPERIMENT_FINDINGS.md` |
| Agentic tool operator | drives Claude Code / Cursor | `CLAUDE.md`, `docs/AGENTIC-WORKFLOW.md` |

### 2.3 Operating environment

| | Requirement |
|---|---|
| OS | Linux (primary), macOS (viewer path), Windows via WSL2 (untested) |
| Python | 3.10+ (3.12 verified) |
| CPU | x86-64; 8 cores comfortable |
| RAM | 4 GB minimum |
| Disk | ~3 GB |
| GPU | **Not required** for Stages 1–2. Required for Stage 3 training. |
| Display | Required only for interactive viewer scripts |

### 2.4 Constraints

- **C1** — Development hardware is CPU-only; GPU access is bursty and remote.
  Every Stage 1–2 feature must be usable without a GPU.
- **C2** — The pretrained policy is a fixed external binary. Its observation
  contract cannot be renegotiated; the system must conform to it exactly.
- **C3** — Policy inference must complete inside the 20 ms budget of a 50 Hz
  control loop.
- **C4** — Results must be reproducible from a clean checkout by a third party.

### 2.5 Assumptions and dependencies

- The `walk-these-ways` checkpoints are obtainable by the operator (§7.2).
- MuJoCo's contact model differs from Isaac Gym's; performance degradation on
  transfer is expected and is the object of study, not a defect.
- The Go2 MJCF from `mujoco_menagerie` faithfully represents the hardware.

---

## 3. Functional requirements — Stage 1: RL fundamentals

| ID | Requirement | Status |
|---|---|---|
| FR-1.1 | Train a PPO agent on CartPole-v1 and reach mean reward ≥ 475 | ✅ ~500 |
| FR-1.2 | Train a PPO agent on LunarLander-v3 and reach mean reward ≥ 200 | ✅ 220–274 |
| FR-1.3 | Train a PPO agent on Pendulum-v1 and reach mean reward ≥ −200 | ✅ met at 300k |
| FR-1.4 | Persist trained models to disk and reload them | ✅ SB3 `.zip` |
| FR-1.5 | Render a trained agent for visual inspection | ✅ `*_watch.py` |
| FR-1.6 | Log training metrics to TensorBoard, comparable across runs | ✅ `tb_logs/<env>/PPO_N` |
| FR-1.7 | Support vectorised environments for sample throughput | ✅ `DummyVecEnv`, 8 envs |
| FR-1.8 | Support both discrete and continuous action spaces | ✅ all three envs |
| FR-1.9 | Evaluate deterministically over ≥ 10 episodes | ✅ `evaluate_policy` |

---

## 4. Functional requirements — Stage 2: Go2 inference

### 4.1 Model handling

| ID | Requirement | Status |
|---|---|---|
| FR-2.1 | Load a Unitree Go2 MJCF and instantiate simulation state | ✅ |
| FR-2.2 | Operate with **no external model checkout** | ✅ vendored `scenes/go2_model/` |
| FR-2.3 | Enumerate bodies, joints, actuators and sensors for inspection | ✅ `02_inspect_go2.py` |
| FR-2.4 | Reset to a known keyframe, then override with the training default pose | ✅ all runtime scripts |
| FR-2.5 | Resolve model and policy locations without machine-specific paths | ✅ `paths.py` |

### 4.2 Observation construction

| ID | Requirement | Status |
|---|---|---|
| FR-2.6 | Construct a 70-dim observation matching the training contract exactly | ✅ verified field-by-field |
| FR-2.7 | Express gravity in the body frame via inverse quaternion rotation | ✅ |
| FR-2.8 | Apply the training per-field scale factors | ✅ `OBS_SCALES` |
| FR-2.9 | Express joint positions relative to the training default pose | ✅ |
| FR-2.10 | Include the two most recent actions in the observation | ✅ |
| FR-2.11 | Generate 4 per-foot clock signals from the gait phase | ✅ |
| FR-2.12 | Maintain a 30-frame rolling observation history | ✅ `deque(maxlen=30)` |
| FR-2.13 | Fail with a shape assertion if the layout is wrong | ✅ `assert obs.shape == (70,)` |

### 4.3 Policy inference

| ID | Requirement | Status |
|---|---|---|
| FR-2.14 | Load TorchScript checkpoints with no training-codebase dependency | ✅ |
| FR-2.15 | Run the adaptation module over the history to obtain an env latent | ✅ |
| FR-2.16 | Concatenate history + latent as the body network input | ✅ (1, 2102) |
| FR-2.17 | Produce 12 joint-position deltas per policy step | ✅ |
| FR-2.18 | Run inference without gradient tracking | ✅ `torch.no_grad()` |
| FR-2.19 | Report an actionable error when checkpoints are absent | ✅ `require_policy()` |

### 4.4 Control

| ID | Requirement | Status |
|---|---|---|
| FR-2.20 | Convert action deltas to joint targets using the training action scale | ✅ 0.25 |
| FR-2.21 | Apply the reduced action scale to hip joints | ✅ ×0.5 at 0,3,6,9 |
| FR-2.22 | Implement a PD torque controller at the physics rate | ✅ Kp 25, Kd 0.6 |
| FR-2.23 | Run the policy at 50 Hz and physics at 500 Hz | ✅ decimation 10 |
| FR-2.24 | Advance the gait clock on the policy clock | ✅ |

### 4.5 Interaction

| ID | Requirement | Status |
|---|---|---|
| FR-2.25 | Visualise the simulation interactively | ✅ `launch_passive` |
| FR-2.26 | Accept live velocity commands from the keyboard | ✅ `07` |
| FR-2.27 | Switch gait (trot/pace/bound) at runtime | ✅ `07`, `08` |
| FR-2.28 | Reset the robot without restarting | ✅ `R` key |
| FR-2.29 | Select the scene from the command line | ✅ `08 <scene.xml>` |
| FR-2.30 | Capture offscreen screenshots for publication | ✅ `08` + imageio |

---

## 5. Functional requirements — experiments

| ID | Requirement | Status |
|---|---|---|
| FR-3.1 | Execute headless trials with no viewer and no real-time pacing | ✅ `harness.py` |
| FR-3.2 | Parameterise a trial by scene, velocity, gait and seed | ✅ |
| FR-3.3 | Randomise initial conditions deterministically from the seed | ✅ `default_rng(seed)` |
| FR-3.4 | Discard a settle window before measurement | ✅ 3 s |
| FR-3.5 | Detect falls by height and tilt thresholds | ✅ 0.15 m / 60° |
| FR-3.6 | Record velocity tracking, drift, height stability and distance | ✅ |
| FR-3.7 | Persist per-trial results as CSV | ✅ |
| FR-3.8 | Generate parametric terrain scenes reproducibly | ✅ verified byte-identical |
| FR-3.9 | Sweep commanded velocity | ✅ Exp 1, 6 × 5 |
| FR-3.10 | Sweep gait | ✅ Exp 2, 3 × 5 |
| FR-3.11 | Sweep terrain | ✅ Exp 3, 10 × 5 |
| FR-3.12 | Distinguish traversal from mere survival | ✅ progress threshold |
| FR-3.13 | Generate publication figures from the CSVs alone | ✅ no policy needed |
| FR-3.14 | Record interpreted findings alongside raw data | ✅ `EXPERIMENT_FINDINGS.md` |
| FR-3.15 | Record run provenance (git SHA, versions, wall time) in results | 📋 not implemented |
| FR-3.16 | Configure sweeps via CLI rather than source edits | 📋 not implemented |

---

## 6. Functional requirements — future stages

### 6.1 Stage 3 — custom policy training

| ID | Requirement | Status |
|---|---|---|
| FR-4.1 | Define a Go2 locomotion environment emitting the 70-dim contract | ✅ `env/go2_env.py` |
| FR-4.2 | Implement a reward function for velocity tracking and gait regularity | ✅ 12 terms |
| FR-4.3 | Apply domain randomisation (friction, mass, motor strength, latency) | ✅ 8 parameters |
| FR-4.4 | Train PPO within a single session's time budget | ✅ implemented; CPU is slow (§7.1) |
| FR-4.5 | Checkpoint and resume across sessions | ✅ optimiser + RNG + curriculum |
| FR-4.6 | Export to TorchScript in the **existing** 70-dim/2102-dim contract | ✅ verified 4 ways |
| FR-4.7 | Evaluate the new policy with the **existing** harness, unmodified | ✅ proven, CI-checked |
| FR-4.8 | Support terrain curricula (motivated by Exp 3's findings) | ✅ 7 levels |
| FR-4.9 | Implement RMA: privileged encoder (phase 1) + distillation (phase 2) | ✅ both phases |
| FR-4.10 | Produce a **converged** policy that beats the baseline | 🔄 needs GPU compute |
| FR-4.11 | MJX/GPU backend for the environment | 📋 designed, not ported |

FR-4.6 and FR-4.7 were the critical ones and are met: a Stage 3 export is
loaded by `paths.load_policy()` and evaluated by `harness.run_trial()` with
zero changes to Stage 2. Three mechanisms enforce this rather than documenting
it — the env imports its constants from `harness.py`, `assert_contract()` runs
before any `.jit` is written, and `tests/test_stage3_contract.py` exercises a
real harness round-trip.

### 6.2 Stage 4 — vision-conditioned locomotion 📋

| ID | Requirement |
|---|---|
| FR-5.1 | Add depth-camera sensors to the MJCF |
| FR-5.2 | Extend the observation space with visual features |
| FR-5.3 | Encode depth images (CNN or learned latent) |
| FR-5.4 | Generate heightfield terrain beyond boxes and ramps |
| FR-5.5 | Demonstrate traversal of terrain the blind policy fails (≥ 5 cm steps) |

### 6.3 Stage 5 — ROS2 deployment 📋

| ID | Requirement |
|---|---|
| FR-6.1 | Wrap inference as a ROS2 node |
| FR-6.2 | Subscribe to velocity commands, publish joint commands |
| FR-6.3 | Interface with Unitree SDK2 |
| FR-6.4 | Meet real-time constraints with bounded jitter |
| FR-6.5 | Provide an emergency stop |

### 6.4 Stage 6 — sim-to-real 📋

| ID | Requirement |
|---|---|
| FR-7.1 | Deploy to physical Go2 hardware |
| FR-7.2 | Measure the sim-to-real gap with the Stage 2 metric set |
| FR-7.3 | Document failure modes |
| FR-7.4 | Iterate domain randomisation from measured gaps |

---

## 7. Non-functional requirements

### 7.1 Performance

| ID | Requirement | Status |
|---|---|---|
| NFR-1.1 | Policy inference < 20 ms/step on CPU (50 Hz budget) | ✅ reported < 2 ms |
| NFR-1.2 | Headless trials run faster than real time | ✅ no pacing, no render |
| NFR-1.3 | Interactive viewer maintains real-time pacing | ✅ sleep-based |
| NFR-1.4 | A full experiment (75 trials) completes in one working session | ✅ |

### 7.2 Reproducibility

| ID | Requirement | Status |
|---|---|---|
| NFR-2.1 | Identical seed ⇒ identical trial results | ✅ `default_rng(seed)` |
| NFR-2.2 | Robot model self-contained in-repo | ✅ |
| NFR-2.3 | Terrain scenes regenerable byte-identically | ✅ verified |
| NFR-2.4 | Figures regenerable from committed CSVs | ✅ verified |
| NFR-2.5 | Python dependencies declared with version bounds | ✅ `requirements.txt` |
| NFR-2.6 | A pinned container image reproduces the environment exactly | ❌ removed — host venv + `requirements.txt` only |
| NFR-2.7 | No machine-specific absolute paths in source | ✅ `paths.py` |
| NFR-2.8 | External policy checkpoints obtainable and their absence diagnosed | ✅ documented + `check_env.py` |

### 7.3 Usability

| ID | Requirement | Status |
|---|---|---|
| NFR-3.1 | A newcomer can verify their environment with one command | ✅ `check_env.py` |
| NFR-3.2 | A newcomer can install everything with one command | ✅ `setup_env.sh` |
| NFR-3.3 | Missing dependencies produce actionable messages | ✅ |
| NFR-3.4 | Scripts are ordered to form a learning progression | ✅ numbered |
| NFR-3.5 | Architecture is documented with diagrams | ✅ `ARCHITECTURE.md` |
| NFR-3.6 | Substantial work is possible without the policy weights | ✅ §7 of DEPENDENCIES |

### 7.4 Maintainability

| ID | Requirement | Status |
|---|---|---|
| NFR-4.1 | Constants centralised or documented where duplicated | ⚠️ still duplicated in `06`–`08`, but drift is now a **test failure** (`tests/test_constants.py`) |
| NFR-4.2 | Automated tests guard the observation contract | ✅ 93 tests; `tests/test_obs_contract.py` pins every field boundary |
| NFR-4.3 | Local verification covers env + tests without a CI host | ✅ `make test`, `scripts/check_env.py`, `train.py --smoke` |
| NFR-4.4 | Style enforced by a linter/formatter | ❌ not configured |
| NFR-4.5 | Agentic coding tools have durable project context | ✅ `CLAUDE.md` |
| NFR-4.6 | Tests runnable without the policy checkpoints | ✅ none of the 93 require them |

### 7.5 Portability

| ID | Requirement | Status |
|---|---|---|
| NFR-5.1 | Runs on any Linux distribution | ✅ multi-distro bootstrap |
| NFR-5.2 | Runs headless (no display) | ✅ `MUJOCO_GL=osmesa` |
| NFR-5.3 | Runs containerised | ❌ removed |
| NFR-5.4 | Runs with no GPU | ✅ CPU-only by design |
| NFR-5.5 | Artefacts written by containers are host-user-owned | ❌ n/a (no containers) |

---

## 8. Data requirements

### 8.1 Trial record schema

Every experiment CSV shares this schema, one row per trial:

| Field | Type | Meaning |
|---|---|---|
| `scene` | str | scene XML filename |
| `cmd_vx`, `cmd_vy`, `cmd_yaw` | float | commanded velocities |
| `gait` | enum | `trot` \| `pace` \| `bound` |
| `seed` | int | RNG seed — determines initial conditions |
| `fell` | bool | did a fall threshold trip |
| `fall_time_s` | float\|null | when, if so |
| `mean_vx`, `mean_vy` | float\|null | mean base velocity over the window |
| `vel_track_err` | float\|null | `abs(cmd_vx − mean_vx)` |
| `lateral_drift` | float\|null | `mean(abs(vy))` |
| `height_mean`, `height_std` | float\|null | posture stability |
| `distance_traveled` | float\|null | straight-line XY displacement |

Fallen trials carry `null` metrics by design — averaging a partial run against
completed runs would corrupt the statistics.

### 8.2 Retention

| Artefact | Committed? | Rationale |
|---|---|---|
| Result CSVs | yes | raw evidence for published claims |
| Findings markdown | yes | the interpretation layer |
| Figures | yes | regenerable, but committed for review |
| Terrain scenes | yes | exact geometry behind the results |
| Robot MJCF + meshes | yes | self-containment |
| Run logs (`*.txt`) | yes | historical record of actual executions |
| TensorBoard logs | no | large, regenerable |
| SB3 model `.zip` | no | large, regenerable |
| Policy `.jit` | no | large external input |

---

## 9. Verification

| Requirement group | How verified |
|---|---|
| FR-1.\* | training runs + TensorBoard curves + `evaluate_policy` scores |
| FR-2.6–2.13 | shape assertions; field-by-field cross-check against training source |
| FR-2.14–2.19 | `05_inspect_policy.py` prints the network I/O contract |
| FR-2.20–2.24 | robot achieves sustained locomotion; Exp 1 100% survival |
| FR-3.\* | 75 committed trial records; regeneration checks |
| NFR-2.\* | scene regeneration `git diff` empty; figures re-rendered |
| NFR-3.\* | `check_env.py` executed on host |
| FR-4.6–4.7 | Stage 3 export loaded by `paths.load_policy()` and run through `harness.run_trial()` |
| FR-4.\* | `tests/test_stage3_contract.py`; `train.py --smoke`; export round-trip |
| NFR-4.2–4.3 | `make test` (93 passing); CI on every push |
| NFR-4.4 | **not verifiable — no linter configured** |

**Verification status:** the observation contract, the model dimensions, the
cross-file constants, scene reproducibility, path resolution and the Stage 3 →
Stage 2 round-trip are now covered by automated tests that need no policy
checkpoints. What remains manual is the *qualitative* claim that the robot
walks well, which is inherently a judgement about a simulation.

---

## 10. Traceability

| Stage | Requirements | Implementation | Evidence |
|---|---|---|---|
| 1 | FR-1.1–1.9 | `stage1-rl-fundamentals/` | `tb_logs/`, README analysis |
| 2 | FR-2.1–2.30 | `stage2-go2-mujoco-inference/01-08` | walking robot, `media/stage2_walking.gif` |
| Research | FR-3.1–3.14 | `experiments/` | `results/*.csv`, `EXPERIMENT_FINDINGS.md` |
| Infra | NFR-2.5–2.8, 3.1–3.3, 5.\* | `scripts/`, `paths.py` | this branch |
| 3 | FR-4.1–4.9 | `stage3-go2-training/` | smoke run + export round-trip + 40 tests |
| 3 | FR-4.10–4.11 | — | blocked on GPU compute |
| Tests | NFR-4.2, 4.6 | `tests/` | 93 passing |
| Local verify | NFR-4.3 | `make test` / `check_env.py` / `--smoke` | run before push |
| 4 | FR-5.\* | — | not implemented |
| 5 | FR-6.\* | — | not implemented |
| 6 | FR-7.\* | — | not implemented |
