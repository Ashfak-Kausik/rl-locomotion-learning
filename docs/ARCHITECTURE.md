# Architecture

A top-down view of what this repository actually is, how the pieces connect,
and where every number comes from. Read this before touching code.

**One-line summary:** this repo takes a neural network that was trained to walk
a robot dog *somewhere else* (NVIDIA Isaac Gym, on a GPU cluster) and makes it
walk *here* (MuJoCo, on a laptop CPU) — then measures, rigorously, how much
gets lost in that translation.

---

## Table of contents

1. [The 30-second version](#1-the-30-second-version)
2. [System context](#2-system-context)
3. [Repository map](#3-repository-map)
4. [Stage 1 — the RL fundamentals subsystem](#4-stage-1--the-rl-fundamentals-subsystem)
5. [Stage 2 — the locomotion inference pipeline](#5-stage-2--the-locomotion-inference-pipeline)
6. [The dual-rate control loop](#6-the-dual-rate-control-loop)
7. [The 70-dimensional observation vector](#7-the-70-dimensional-observation-vector)
8. [The policy network](#8-the-policy-network)
9. [State layout and index conventions](#9-state-layout-and-index-conventions)
10. [The experiment subsystem](#10-the-experiment-subsystem)
11. [Runtime & deployment topology](#11-runtime--deployment-topology)
12. [Where the numbers live](#12-where-the-numbers-live)
13. [Stage 3 closes the loop](#13-stage-3-closes-the-loop)

---

## 1. The 30-second version

```
     TRAINING (elsewhere, not in this repo)          THIS REPO
   ┌────────────────────────────────────┐    ┌──────────────────────────────┐
   │  walk-these-ways (MIT)             │    │  Load .jit checkpoints       │
   │  Isaac Gym · GPU · 4096 parallel   │───►│  Rebuild the observation     │
   │  envs · PPO · domain randomization │    │  vector by hand from MuJoCo  │
   │                                    │    │  Drive a PD controller       │
   │  OUTPUT: two TorchScript files     │    │  Measure what breaks         │
   └────────────────────────────────────┘    └──────────────────────────────┘
              body_latest.jit                          ▲
              adaptation_module_latest.jit  ────────────┘
```

The hard part is not "run a neural network". The hard part is that the policy
expects its inputs in an *exact* format — a specific order, specific scale
factors, a specific default joint pose, specific sign conventions — none of
which are written down anywhere. All of it had to be recovered by reading the
training source. Get one sign wrong and the robot flops over.

---

## 2. System context

```
                                  ┌───────────────────────────────────┐
                                  │        HUMAN / RESEARCHER         │
                                  └───────────────┬───────────────────┘
                     commands (vx, vy, yaw, gait) │  ▲  metrics, plots, video
                                                  ▼  │
   ┌──────────────────────────────────────────────────────────────────────┐
   │                    rl-locomotion-learning (this repo)                │
   │                                                                      │
   │   ┌────────────────────┐          ┌──────────────────────────────┐   │
   │   │  Stage 1           │          │  Stage 2                     │   │
   │   │  RL fundamentals   │          │  Go2 locomotion inference    │   │
   │   │                    │          │                              │   │
   │   │  Gymnasium envs    │          │  MuJoCo + MJCF scenes        │   │
   │   │  SB3 PPO (train)   │          │  TorchScript policy (infer)  │   │
   │   │  TensorBoard       │          │  Experiment harness → CSV    │   │
   │   └────────────────────┘          └──────────────────────────────┘   │
   │            │                                    │                    │
   │            │  writes                            │  writes            │
   │            ▼                                    ▼                    │
   │      tb_logs/, *.zip                  results/*.csv, figures/*.png   │
   └──────────────────────────────────────────────────────────────────────┘
                    ▲                                   ▲
        pip packages│                                   │ external artefacts
   ┌────────────────┴───────────┐         ┌─────────────┴────────────────┐
   │ gymnasium · stable-        │         │ walk-these-ways .jit weights │
   │ baselines3 · mujoco ·      │         │   (NOT in repo — you supply) │
   │ torch · numpy · matplotlib │         │ go2 MJCF + meshes            │
   └────────────────────────────┘         │   (vendored in repo ✓)       │
                                          └──────────────────────────────┘
```

**Trust boundary worth noticing:** everything inside the repo is reproducible
from a clean checkout *except* the two `.jit` policy files. That is the single
external dependency that cannot be `pip install`ed. See
[DEPENDENCIES.md](DEPENDENCIES.md).

---

## 3. Repository map

```
rl-locomotion-learning/
│
├── README.md                        Project overview, results, stage roadmap
├── requirements.txt                 All Python deps  ← added by onboarding work
├── Makefile                         Convenience targets (make help)
├── CLAUDE.md                        Context file read by agentic coding tools
│
├── reinforcement-learning-theories/ 8 chapters, ~1100 lines, written for this
│   ├── chapter-01-introduction.md       project. Chapters 1-5 = RL theory,
│   ├── ...                              6-7 = robotics/locomotion,
│   └── chapter-08-this-project-in-detail  8 = a run-by-run account of the work
│
├── stage1-rl-fundamentals/          ✅ COMPLETE — PPO on 3 classic benchmarks
│   ├── 01_cartpole_ppo.py           train    ┐
│   ├── 02_cartpole_watch.py         evaluate │  the pattern repeats:
│   ├── 03_lunarlander_ppo.py        train    │  odd script trains + saves,
│   ├── 04_lunarlander_watch.py      evaluate │  even script loads + renders
│   ├── 05_pendulum_ppo.py           train    │
│   ├── 06_pendulum_watch.py         evaluate ┘
│   ├── tb_logs/                     TensorBoard event files (gitignored)
│   └── README.md                    Detailed metric-by-metric write-up
│
├── stage2-go2-mujoco-inference/     ✅ COMPLETE — the core of the project
│   ├── paths.py                     Env-overridable path resolution ← added
│   ├── 01_hello_go2.py              load MJCF, open viewer          ⎫
│   ├── 02_inspect_go2.py            enumerate bodies/joints/actuators⎪ build-up,
│   ├── 03_pose_go2.py               hand-written PD stands the robot ⎬ each one
│   ├── 04_build_obs_vector.py       first obs attempt (SUPERSEDED)   ⎪ adds one
│   ├── 05_inspect_policy.py         probe .jit input/output shapes   ⎪ concept
│   ├── 06_run_policy.py             ★ full pipeline — robot walks    ⎭
│   ├── 07_run_policy_interactive.py keyboard teleop + gait switching
│   ├── 08_view_scene.py             scene-selectable viewer for screenshots
│   │
│   ├── scenes/                      Self-contained MuJoCo worlds
│   │   ├── go2_model/               vendored Unitree Go2 MJCF + 16 .obj meshes
│   │   ├── go2_flat.xml             flat ground
│   │   ├── go2_slope_{05..25}.xml   inclines 5/10/15/20/25°
│   │   └── go2_stairs_{02..16}.xml  steps 2/5/8/12/16 cm
│   │
│   ├── experiments/                 The research layer
│   │   ├── harness.py               ★ headless run_trial() — shared backbone
│   │   ├── generate_terrain_scenes.py  writes the slope/stair XMLs above
│   │   ├── check_terrain_visual.py     eyeball a scene, no policy needed
│   │   ├── exp1_velocity_sweep.py      6 velocities × 5 seeds
│   │   ├── exp2_gait_robustness.py     3 gaits × 5 seeds
│   │   ├── exp3_terrain.py             10 terrains × 5 seeds
│   │   ├── make_figures.py             CSV → 4 paper figures
│   │   ├── results/                    CSVs + EXPERIMENT_FINDINGS.md
│   │   └── figures/                    generated PNGs
│   │
│   └── paper_figures/final_figures/ 11 curated PNGs for the write-up
│
├── stage3-go2-training/             ⚙️ COMPLETE pipeline — awaiting GPU compute
│   ├── networks.py                  ★ THE CONTRACT + RMA architecture
│   ├── config.py                    hyperparameters + reward weights
│   ├── train.py                     PPO (phase 1) + distillation (phase 2)
│   ├── export.py                    checkpoint → TorchScript, verified 4 ways
│   ├── evaluate.py                  exported policy vs baseline, via harness
│   └── env/
│       ├── go2_env.py               emits the exact 70-dim observation
│       ├── rewards.py               12 terms
│       ├── curriculum.py            7 levels, seeded by Experiment 3
│       └── domain_rand.py           8 params = the RMA privileged vector
│
├── tests/                           93 tests, none need policy weights
│   ├── test_obs_contract.py         ★ defends the 70-dim layout
│   ├── test_model.py                MJCF structure
│   ├── test_constants.py            cross-file drift guard
│   ├── test_scenes_and_paths.py     reproducibility + path resolution
│   └── test_stage3_contract.py      Stage 3 → Stage 2 round-trip
│
├── docs/                            ← this documentation set
├── scripts/                         ← setup_env.sh, check_env.py, hw_profile.py
├── hardware/profiles/               ← committed per-machine CPU/GPU records
└── media/                           Demo GIFs used by the READMEs
```

**The numbered-script convention is the most important thing to internalise.**
Both stages use it. Script `NN` assumes you understand script `NN-1`. They are
a curriculum, not an application — which is why there is duplicated code
between `06`, `07`, `08` and `harness.py`. See
[REVERSE-ENGINEERING.md](REVERSE-ENGINEERING.md#the-duplication-question).

---

## 4. Stage 1 — the RL fundamentals subsystem

Stage 1 is a conventional, library-driven RL setup. Almost no custom code —
that is the point. It exists to build intuition before Stage 2 removes all the
safety rails.

```
  ┌──────────────┐   obs    ┌─────────────────┐  action  ┌──────────────┐
  │ Gymnasium    │─────────►│ SB3 PPO         │─────────►│ Gymnasium    │
  │ environment  │          │ "MlpPolicy"     │          │ environment  │
  │              │◄─────────│ actor + critic  │◄─────────│  .step()     │
  └──────────────┘  reward  └────────┬────────┘          └──────────────┘
    CartPole-v1                      │
    LunarLander-v3                   │ scalar metrics per rollout
    Pendulum-v1                      ▼
                            ┌─────────────────┐      ┌──────────────────┐
                            │ TensorBoard log │─────►│ tb_logs/<env>/   │
                            └─────────────────┘      │   PPO_1, PPO_2…  │
                                     │               └──────────────────┘
                                     ▼
                            model.save("ppo_cartpole") → .zip
                                     │
                                     ▼
                            02/04/06_*_watch.py  →  render_mode="human"
```

Environment progression, and why it is in that order:

| # | Env | Obs dim | Action space | Concept it introduces |
|---|-----|---------|--------------|-----------------------|
| 1 | CartPole-v1 | 4 | discrete (2) | the train→save→evaluate loop; reading TensorBoard |
| 2 | LunarLander-v3 | 8 | discrete (4) | shaped rewards; vectorised envs; **policy collapse** |
| 3 | Pendulum-v1 | 3 | **continuous (1)** | Gaussian policies, `train/std` — the prerequisite for joint control |

Continuous actions are the bridge: a quadruped needs 12 real-valued joint
targets, which is Pendulum's 1-D problem scaled up.

---

## 5. Stage 2 — the locomotion inference pipeline

This is the system. Every arrow is a place a bug can hide.

```
 ┌────────────────────────────────────────────────────────────────────────────┐
 │                        ONE POLICY STEP  (runs at 50 Hz)                    │
 └────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────┐
  │  MuJoCo MjData       │   the ground-truth simulator state
  │  qpos[19] qvel[18]   │
  └──────────┬───────────┘
             │
      ┌──────┴───────────────────────────────────────────────┐
      │ extract & transform  (harness.py :: build_obs)        │
      │                                                       │
      │  qpos[3:7]  quaternion ──► quat_rotate_inverse ──► projected gravity (3)
      │  COMMANDS   15 values   ──► × COMMANDS_SCALE   ──► scaled commands (15)
      │  qpos[7:19] joint pos   ──► − DEFAULT_JOINT_POS ──► joint pos rel   (12)
      │  qvel[6:18] joint vel   ──► × 0.05              ──► joint vel       (12)
      │  last_action                                    ──► action t−1     (12)
      │  prev_action                                    ──► action t−2     (12)
      │  gait_phase_t ──► 4 per-foot phases ──► sin()   ──► clock signals   (4)
      └──────────────────────────┬────────────────────────────┘
                                 │  concatenate  →  obs (70,)
                                 ▼
      ┌───────────────────────────────────────────────┐
      │  obs_history : deque(maxlen=30)               │   ← the robot's memory
      │  [t−29][t−28] … [t−1][t]                      │
      │  flatten → (1, 2100)                          │
      └──────────────┬────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         │
  ┌───────────────────┐           │
  │ adaptation_module │           │   "what kind of world am I in?"
  │  (1,2100)→(1,2)   │           │   infers latent terrain/dynamics
  └─────────┬─────────┘           │   properties it cannot directly see
            │ env latent (1,2)    │
            └──────────┬──────────┘
                       │ torch.cat → (1, 2102)
                       ▼
             ┌───────────────────┐
             │   body (actor)    │   the walking policy itself
             │  (1,2102)→(1,12)  │
             └─────────┬─────────┘
                       │ action_delta (12,)  — dimensionless, roughly ±1
                       ▼
      ┌────────────────────────────────────────────────────┐
      │ joint_targets = DEFAULT_JOINT_POS                  │
      │                 + action_delta × action_scale      │
      │                                                    │
      │   action_scale = 0.25 everywhere,                  │
      │                  ×0.5 extra on the 4 hip joints    │
      └────────────────────┬───────────────────────────────┘
                           │  joint_targets (12,) in radians
                           ▼
 ┌────────────────────────────────────────────────────────────────────────────┐
 │              PD CONTROL  (runs at 500 Hz — 10× faster, see §6)             │
 │                                                                            │
 │     τ = Kp·(joint_targets − qpos[7:19])  −  Kd·qvel[6:18]                   │
 │         Kp = 25.0                            Kd = 0.6                      │
 │                                                                            │
 │     data.ctrl[:] = τ        ← the Go2 MJCF exposes TORQUE motors,          │
 │                               not position servos. This PD loop is         │
 │                               the thing that turns an angle into a force.  │
 └────────────────────────────────┬───────────────────────────────────────────┘
                                  ▼
                        ┌───────────────────┐
                        │ mujoco.mj_step()  │  advance physics 2 ms
                        └─────────┬─────────┘
                                  │
                                  └──────────► back to MjData, loop
```

### Why "adaptation module"

This is the **RMA** (Rapid Motor Adaptation) architecture. During training a
privileged network could see things the real robot cannot — friction
coefficients, payload mass, motor strength — and compressed them into a small
latent vector. The adaptation module is a student network that learns to guess
that same latent from *proprioceptive history alone*. That is why the policy
needs 30 frames of history instead of just the current state: the latent is
inferred from how the robot has been responding over the last 0.6 seconds.

---

## 6. The dual-rate control loop

The single most common sim-to-sim bug. The policy was trained at 50 Hz. MuJoCo
runs the Go2 model at 500 Hz. If you call the policy every physics step, it
sees the world moving 10× too slowly and the gait falls apart.

```
  physics step  0     1     2     3     4     5     6     7     8     9    10
                │     │     │     │     │     │     │     │     │     │     │
  time (ms)     0     2     4     6     8    10    12    14    16    18    20
                │                                                           │
  policy       ███                                                         ███
  (50 Hz)     infer                                                       infer
                │                                                           │
  PD control   ███   ███   ███   ███   ███   ███   ███   ███   ███   ███   ███
  (500 Hz)      τ     τ     τ     τ     τ     τ     τ     τ     τ     τ     τ
                └─────────────── same joint_targets held ──────────────────┘

  DECIMATION = 10        model.opt.timestep = 0.002 s
  policy rate = 1 / (0.002 × 10) = 50 Hz          ✓ matches training
```

In code this is the entire mechanism:

```python
if sim_step % DECIMATION == 0:      # 50 Hz  — expensive: neural network
    joint_targets = policy(...)
data.ctrl[:] = KP * (joint_targets - q) - KD * qd   # 500 Hz — cheap: arithmetic
mujoco.mj_step(model, data)
```

The gait clock advances on the **policy** clock, not the physics clock:

```python
gait_phase_t = (gait_phase_t + STEP_FREQUENCY * dt * DECIMATION) % 1.0
#                               2.0 Hz         0.002s     10
```

---

## 7. The 70-dimensional observation vector

Recovered field-by-field from the `walk-these-ways` training source. **Order
matters absolutely** — the network has no names for these, only positions.

```
 idx  ┌────────────────────────────────────────────────┬──────┬──────────────┐
      │ field                                          │ dims │ scale        │
 ═════╪════════════════════════════════════════════════╪══════╪══════════════╡
 0    │ projected gravity (body frame)                 │  3   │ —            │
      │   which way is "down" from the robot's POV;    │      │              │
      │   ≈ [0,0,−1] upright. This is the whole IMU.   │      │              │
 ─────┼────────────────────────────────────────────────┼──────┼──────────────┤
 3    │ cmd lin_vel_x                                  │  1   │ × 2.0        │
 4    │ cmd lin_vel_y                                  │  1   │ × 2.0        │
 5    │ cmd ang_vel_yaw                                │  1   │ × 0.25       │
 6    │ cmd body_height                                │  1   │ × 2.0        │
 7    │ cmd step_frequency        (Hz)                 │  1   │ × 1.0        │
 8    │ cmd gait_phase       ┐                         │  1   │ × 1.0        │
 9    │ cmd gait_offset      ├ the gait selector       │  1   │ × 1.0        │
 10   │ cmd gait_bound       ┘                         │  1   │ × 1.0        │
 11   │ cmd gait_duration         (stance fraction)    │  1   │ × 1.0        │
 12   │ cmd footswing_height (m)                       │  1   │ × 0.15       │
 13   │ cmd body_pitch       (rad)                     │  1   │ × 0.3        │
 14   │ cmd body_roll        (rad)                     │  1   │ × 0.3        │
 15   │ cmd stance_width     (m)                       │  1   │ × 1.0        │
 16   │ cmd stance_length    (m)                       │  1   │ × 1.0        │
 17   │ cmd aux_reward                                 │  1   │ × 1.0        │
 ─────┼────────────────────────────────────────────────┼──────┼──────────────┤
 18   │ (joint_pos − DEFAULT_JOINT_POS)                │ 12   │ × 1.0        │
 30   │ joint_vel                                      │ 12   │ × 0.05       │
 42   │ last_action     (action at t−1)                │ 12   │ —            │
 54   │ prev_action     (action at t−2)                │ 12   │ —            │
 ─────┼────────────────────────────────────────────────┼──────┼──────────────┤
 66   │ clock = sin(2π · foot_phase) for 4 feet        │  4   │ —            │
 ═════╧════════════════════════════════════════════════╧══════╧══════════════╡
                                                    TOTAL = 70
```

### The scale factors are not cosmetic

Neural networks train badly when inputs span wildly different magnitudes.
Joint velocities reach ±20 rad/s while joint positions sit near ±1 rad, so
velocities are multiplied by `0.05` to bring both into a comparable band. These
exact constants were used at training time, so inference **must** reproduce
them — a wrong scale is functionally the same as feeding the network noise.

### The gait clock

Gait selection is not a discrete mode switch. Three continuous phase-offset
commands reshape *when* each foot is told to be in swing:

```
  foot_phases[0] = t + phase + offset + bound     ← FR
  foot_phases[1] = t + offset                     ← FL
  foot_phases[2] = t + bound                      ← RR
  foot_phases[3] = t + phase                      ← RL
  clock = sin(2π · (foot_phases mod 1))

  GAIT_PRESETS = {(phase, offset, bound)}
    trot  (0.5, 0.0, 0.0)   diagonal pairs move together
    pace  (0.0, 0.5, 0.0)   left pair / right pair
    bound (0.0, 0.0, 0.5)   front pair / rear pair
```

Same network, same weights — only these three numbers change. That is what
"gait-conditioned" means, and it is `walk-these-ways`'s central contribution.

> **A superseded version of this layout still lives in the repo.**
> `04_build_obs_vector.py` uses a different (incorrect) field ordering and
> flipped hip signs. It is a preserved learning artefact from before the layout
> was recovered correctly. **Never copy from it** — copy from `harness.py`.

---

## 8. The policy network

```
                  obs history (1, 2100)
                   30 frames × 70 dims
                          │
            ┌─────────────┴─────────────┐
            │                           │
            ▼                           │
   ┌──────────────────┐                 │
   │ adaptation_module│                 │
   │  MLP             │                 │
   │  2100 → … → 2    │                 │
   └────────┬─────────┘                 │
            │ (1, 2)                    │
            └────────────┬──────────────┘
                         │ concat
                         ▼
                   (1, 2102)
                         │
                ┌────────┴────────┐
                │  body / actor   │
                │  MLP            │
                │  2102 → … → 12  │
                └────────┬────────┘
                         ▼
              action deltas (1, 12)
```

Both arrive as **TorchScript** (`.jit`) — a serialised, self-contained graph.
No Python class definition is needed to load them, which is exactly why they
survive the move from the Isaac Gym codebase to this repo. `05_inspect_policy.py`
exists purely to print these shapes and confirm the contract.

Inference cost: **< 2 ms per step on CPU**, comfortably inside the 20 ms budget
a 50 Hz loop allows.

---

## 9. State layout and index conventions

MuJoCo hands you two flat arrays. Slicing them correctly is half the work.

```
  qpos  (19 values)                       qvel  (18 values)
  ┌────────────────────────────┐          ┌────────────────────────────┐
  │ [0:3]   base position xyz  │          │ [0:3]   base lin. velocity │
  │ [3:7]   base quaternion    │  ← 4     │ [3:6]   base ang. velocity │  ← 3
  │         (w, x, y, z)       │          │                            │
  │ [7:19]  12 joint angles    │          │ [6:18]  12 joint velocities│
  └────────────────────────────┘          └────────────────────────────┘

  The offset differs (7 vs 6) because a free-floating body needs 4 numbers
  for orientation in position space but only 3 in velocity space.
  Getting this wrong is the classic first bug.
```

Joint / actuator ordering, verified against `scenes/go2_model/go2.xml`:

```
  index:   0      1       2      3      4       5      6      7       8      9     10      11
         FL_hip FL_thigh FL_calf FR_hip FR_thigh FR_calf RL_hip RL_thigh RL_calf RR_hip RR_thigh RR_calf
         └──── front left ────┘ └──── front right ───┘ └──── rear left ────┘ └──── rear right ───┘

  hip joints are at indices 0, 3, 6, 9   (these get the ×0.5 action scale)

  DEFAULT_JOINT_POS (training values, NOT the MJCF keyframe):
     FL:  +0.1, 0.8, −1.5        ┐ hip sign is POSITIVE on the left side,
     FR:  −0.1, 0.8, −1.5        │ NEGATIVE on the right — a mirror
     RL:  +0.1, 1.0, −1.5        │ convention that had to be recovered
     RR:  −0.1, 1.0, −1.5        ┘ from the training config
```

**Two poses exist and they are not the same.** The MJCF `home` keyframe is
`0, 0.9, −1.8` per leg. The training default is `±0.1, 0.8/1.0, −1.5`. Every
script resets to the keyframe and then *overwrites* the joint block:

```python
mujoco.mj_resetDataKeyframe(model, data, key_id)   # sets a valid full state
data.qpos[7:19] = DEFAULT_JOINT_POS                # then force training pose
data.qpos[2] = 0.30                                # and training body height
mujoco.mj_forward(model, data)                     # recompute derived state
```

Skip that override and the observation vector is offset by a constant on all
12 joints, which the policy reads as "my legs are in a pose I have never seen".

---

## 10. The experiment subsystem

Stage 2 turns from a demo into research here. `harness.py` is the shared core;
each experiment is a thin sweep over it.

```
  ┌────────────────────────────┐
  │ generate_terrain_scenes.py │  parametric MJCF writer, run once
  └─────────────┬──────────────┘
                │ writes 10 XMLs (committed, so terrain is reproducible)
                ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  scenes/  go2_flat · go2_slope_{05,10,15,20,25} ·                 │
  │           go2_stairs_{02,05,08,12,16}                             │
  └─────────────┬────────────────────────────────────────────────────┘
                │
                ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  harness.py :: run_trial(scene, vx, vy, yaw, gait, seed, …)      │
  │                                                                  │
  │   ① load scene, reset to keyframe, apply training default pose   │
  │   ② seeded initial-condition randomisation  ← makes trials       │
  │        joints ±0.02 rad · height ±0.01 m ·     statistically      │
  │        yaw ±0.05 rad · joint vel ±0.05        independent        │
  │   ③ settle window (3 s, discarded)                               │
  │   ④ measurement window (20–30 s, logged)                         │
  │   ⑤ fall check every physics step:                               │
  │        height < 0.15 m  OR  tilt > 60°  →  abort, record time    │
  │   ⑥ return metrics dict                                          │
  └─────────────┬────────────────────────────────────────────────────┘
                │  one dict per trial
    ┌───────────┼────────────┬─────────────────┐
    ▼           ▼            ▼                 │
 exp1        exp2         exp3                 │  NO viewer, NO sleep —
 velocity    gait         terrain              │  runs as fast as the CPU
 6×5 trials  3×5 trials   10×5 trials          │  allows
    │           │            │                 │
    └───────────┴────────────┴─────────────────┘
                │  csv.DictWriter
                ▼
  ┌──────────────────────────────────┐
  │ results/exp{1,2,3}_*.csv         │  ← committed: raw evidence
  │ results/EXPERIMENT_FINDINGS.md   │  ← committed: interpretation
  └─────────────┬────────────────────┘
                ▼
  ┌──────────────────────────────────┐
  │ make_figures.py (matplotlib Agg) │  ← runs with NO policy weights,
  │  → figures/fig{1,2,3,4}_*.png    │     straight from the CSVs
  └──────────────────────────────────┘
```

### Metrics computed per trial

| Field | Meaning |
|---|---|
| `fell`, `fall_time_s` | did the height/tilt threshold trip, and when |
| `mean_vx`, `mean_vy` | mean base velocity over the measurement window |
| `vel_track_err` | `abs(commanded_vx − mean_vx)` — the headline sim-to-sim gap |
| `lateral_drift` | `mean(abs(vy))` — sideways wander |
| `height_mean`, `height_std` | posture stability |
| `distance_traveled` | straight-line XY displacement across the window |

`distance_traveled` is the one that mattered most: it exposed that a robot can
score 100% survival on stairs by *safely stalling* at the first step. The
traversability metric was corrected mid-study to require survival **and**
> 3.5 m progress. That correction is documented in `EXPERIMENT_FINDINGS.md`
and is a genuine methodological result, not an afterthought.

---

## 11. Runtime & deployment topology

Three ways to run, all producing identical numbers:

```
  ┌─ Host virtualenv ─────────────────────────────────────────────────┐
  │    ./scripts/setup_env.sh  →  .venv/  →  source .venv/bin/activate│
  │    Best for: interactive viewer work, day-to-day development      │
  │    Rendering: native GLFW, or MUJOCO_GL=egl / osmesa headless     │
  └───────────────────────────────────────────────────────────────────┘

  Path resolution — paths.py reads:
      GO2_SCENE / GO2_MODEL_PATH   which world to load
      GO2_POLICY_DIR               where the .jit checkpoints live
```

---

## 12. Where the numbers live

When you need to change a constant, this is the authoritative location. Note
that several constants are **duplicated** across scripts by design (each script
is meant to be readable standalone) — change one, grep for the rest.

| Constant | Value | Defined in | Meaning |
|---|---|---|---|
| `DECIMATION` | 10 | `harness.py`, `06`, `07`, `08` | physics steps per policy step |
| `model.opt.timestep` | 0.002 s | `scenes/go2_model/go2.xml` | physics rate (500 Hz) |
| `HISTORY_LEN` | 30 | `harness.py`, `06`, `07`, `08` | frames the adaptation module sees |
| `OBS_DIM` | 70 | same | observation width |
| `ACTION_SCALE` | 0.25 | same | joint-delta magnitude |
| `HIP_SCALE_REDUCTION` | 0.5 | same | extra damping on hip abduction |
| `KP` / `KD` | 25.0 / 0.6 | same | PD gains for the 500 Hz torque loop |
| `DEFAULT_JOINT_POS` | see §9 | same | training default pose |
| `OBS_SCALES` | see §7 | same | per-field input normalisation |
| `GAIT_PRESETS` | see §7 | `harness.py`, `07`, `08` | trot / pace / bound |
| `STEP_FREQUENCY` | 2.0 Hz | `harness.py` | gait clock rate |
| `FALL_HEIGHT_THRESHOLD` | 0.15 m | `harness.py` | ≈ half nominal standing height |
| `FALL_TILT_THRESHOLD_DEG` | 60° | `harness.py` | past self-recovery |
| `PROGRESS_THRESHOLD_M` | 3.5 m | `exp3_terrain.py` | run-up 2.5 m + ≥1 m real progress |
| `POLICY_DIR`, `MODEL_PATH` | env-overridable | **`paths.py`** | external artefact locations |

---

## 13. Stage 3 closes the loop

Stage 2 measured a baseline. Stage 3 trains a replacement — and the design rule
that makes that worth doing is that **the new policy plugs into the old
measurement apparatus unchanged**:

```
   stage3-go2-training/                     stage2-go2-mujoco-inference/
   ┌────────────────────────┐               ┌──────────────────────────────┐
   │ Go2Env                 │  imports its  │ harness.py                   │
   │   build_obs ───────────┼──────────────►│   DEFAULT_JOINT_POS, KP/KD,  │
   │   DECIMATION, KP, KD   │  constants    │   DECIMATION, build_obs, …   │
   └───────────┬────────────┘   FROM ──────►└──────────────────────────────┘
               │ trains                                    ▲
               ▼                                           │
   ┌────────────────────────┐                              │
   │ export.py              │  body_latest.jit             │
   │   assert_contract()    │─ adaptation_module_latest.jit┤
   │   verify 4 ways        │                              │
   └────────────────────────┘                              │
                                    GO2_POLICY_DIR ────────┘
                                    (paths.py resolves it)

   Result: run_trial(), exp1/2/3 and make_figures.py all work on the new
   policy with ZERO code changes — same scenes, same seeds, same metrics,
   directly comparable against the walk-these-ways baseline.
```

Three mechanisms enforce this rather than merely documenting it:

1. `env/go2_env.py` **imports** its constants from `harness.py` — it never
   redeclares them, so drift is impossible by construction.
2. `networks.assert_contract()` runs before `export.py` writes any file.
3. `tests/test_stage3_contract.py` performs a real `run_trial()` round-trip on
   an exported policy; CI runs it on every push.

---

## Related reading

- [`../stage3-go2-training/README.md`](../stage3-go2-training/README.md) — the training stack in detail
- [TECH-STACK-PRIMER.md](TECH-STACK-PRIMER.md) — the languages, libraries and APIs used above
- [REVERSE-ENGINEERING.md](REVERSE-ENGINEERING.md) — what is implemented, what is broken, what is next
- [TDD.md](TDD.md) — component-level design decisions and rationale
- [SRS.md](SRS.md) — formal requirements
- `../reinforcement-learning-theories/` — the theory behind all of it
