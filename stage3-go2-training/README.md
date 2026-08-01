# Stage 3 — Custom Go2 Policy Training

Train a locomotion policy from scratch that Stage 2's experiment harness can
evaluate **without modification**, so the result is directly comparable against
the `walk-these-ways` baseline.

```bash
# 30-second end-to-end sanity check — both RMA phases, tiny budgets
python train.py --smoke

# a real run
python train.py --run-name v1 --timesteps 5000000

# resume after a session dies (free-tier GPUs die often)
python train.py --resume runs/v1/checkpoint_latest.pt

# export to TorchScript in the Stage 2 contract
python export.py --checkpoint runs/v1/checkpoint_final.pt

# evaluate against the baseline using Stage 2's harness, unmodified
export GO2_POLICY_DIR=../policies/stage3-v1
python evaluate.py --terrain
```

---

## The one rule

Everything here exists to satisfy a single interface:

```
    adaptation_module :  (B, 2100)  ->  (B, 2)
    body              :  (B, 2102)  ->  (B, 12)

    2100 = 30 history frames x 70 observation dims
    2102 = 2100 + 2 environment-latent dims
      12 = joint position deltas
```

Honour it and `stage2-go2-mujoco-inference/experiments/harness.py` evaluates a
Stage 3 policy on the same scenes, with the same seeded initial conditions, the
same fall criteria and the same metrics as the baseline. Break it and every
measurement Stage 2 produced becomes useless for comparison.

Three things enforce this rather than merely documenting it:

| Mechanism | Where |
|---|---|
| The env **imports** its constants from `harness.py`, never redeclares them | `env/go2_env.py` |
| `assert_contract()` runs before any `.jit` file is written | `networks.py`, `export.py` |
| 40 tests, no checkpoint required, including a real harness round-trip | `tests/test_stage3_contract.py` |

---

## What is here

```
stage3-go2-training/
├── networks.py         the contract + RMA architecture
├── config.py           all hyperparameters and reward weights
├── train.py            PPO (phase 1) + adaptation distillation (phase 2)
├── export.py           checkpoint -> TorchScript, verified 4 ways
├── evaluate.py         exported policy vs baseline, via Stage 2's harness
└── env/
    ├── go2_env.py      the environment — emits the exact 70-dim observation
    ├── rewards.py      12 reward terms
    ├── curriculum.py   terrain schedule derived from Experiment 3
    └── domain_rand.py  8 randomised parameters = the privileged vector
```

---

## RMA in two phases

The deployed robot cannot measure its own friction coefficient. Rapid Motor
Adaptation solves this by training a teacher that can, then distilling a
student that infers the same thing from proprioceptive history.

```
  PHASE 1 — PPO
  ─────────────
     true sim params ──► PrivilegedEncoder ──► latent (2)
     (friction, mass,          [teacher]              │
      motor strength…)                                ▼
     obs history (2100) ─────────────────────► [ Body ] ──► 12 actions
                                                     ▲
                                              Critic ┘  (value, discarded)

  PHASE 2 — supervised distillation
  ─────────────────────────────────
     obs history (2100) ──► AdaptationModule ──► latent (2)
                                [student]            │
                                                     ▼
                                        regress onto the teacher's latent
                                        (MSE, on-policy data)

  EXPORTED:  AdaptationModule + Body       ← the only two that deploy
  DISCARDED: PrivilegedEncoder + Critic    ← training scaffolding
```

Phase 2 collects data **on-policy** with the frozen phase-1 policy, so the
history distribution matches what the deployed policy actually sees. Training
the student on off-distribution data is the classic way to get a module that
works in the lab and fails on the robot.

---

## Design decisions driven by measurement

Stage 2 did not just produce a baseline — it produced numbers that tell you
what to fix. Three of them shaped this code directly.

### Exponential tracking kernel, not squared error

Experiment 1 found the baseline systematically **under-tracks**: 37–55% of
commanded velocity, saturating around 0.55 m/s. A squared-error penalty
saturates too and stops pushing once the error is large — exactly the wrong
shape when the failure mode is "consistently too slow". The exponential kernel
`exp(-err²/σ²)` keeps a usable gradient all the way in.

### Lateral drift gets its own reward term

Experiment 1 finding F1.3: drift is *non-monotonic* in commanded speed, with a
pronounced minimum at 1.0 m/s and higher drift on either side. That is the
signature of a policy that never had drift penalised directly.

### The curriculum is Experiment 3's failure boundary

Not guesswork — the measured points at which the flat-trained baseline breaks:

| Level | Scene | Baseline result (Exp 3 / F1.2) |
|---|---|---|
| 0 `flat-slow` | flat | trivial |
| 1 `flat-fast` | flat | full command range, ≤1.0 m/s |
| 2 `flat-run` | flat | **beyond baseline**: cmd up to 2.5 m/s, past its ~0.55 m/s ceiling (F1.2) |
| 3 `slope-10` | 10° | 100% survival, traversable |
| **4 `slope-15`** | 15° | **20% survival — baseline breaks here** |
| 5 `stairs-5` | 5 cm | **safe stall**: 0.47 m, never climbs |
| 6 `slope-20` | 20° | 0% survival |
| 7 `stairs-8` | 8 cm | far beyond baseline |
| 8 `obstacles-easy` | 12 scattered boxes, 10–20 cm | baseline handles it: 7.53m in 25s, height never drops (verified) |
| 9 `obstacles-hard` | 24 scattered boxes, 15–35 cm | not yet evaluated |
| 10 `gauntlet` | 5cm stairs into a 30-obstacle field | baseline safe-stalls at the first step: 1.34m in 25s (verified, same failure mode as F3.4) |

Levels 0–3 reproduce or exceed the baseline's flat-ground competence (`flat-run`
asks for genuine running speed, not just a longer fast-trot). **Level 4
onward is where a new policy has to beat the baseline on terrain it cannot
handle** — `BASELINE_CEILING = 4` marks that line.

The obstacle levels are **blind**: no camera, no elevation map, proprioception
only — consistent with the 70-dim contract this whole repo is built around,
but a real limitation. State-of-the-art obstacle traversal (e.g. robot
parkour, perceptive locomotion work cited below) uses depth or height-map
input specifically because blind obstacle avoidance is much harder. Treat
`obstacles-*` as "can it recover from an unseen bump/step," not "can it see
and route around a field" — those are different problems.

Promotion needs recent mean return ≥ 75% of the achievable maximum; sustained
failure demotes. Demotion matters: without it, an agent pushed past its
competence collapses with no way back.

---

## Domain randomisation

Eight parameters, resampled per episode. Their normalised values *are* the
privileged vector the teacher sees.

| Parameter | Range | Why |
|---|---|---|
| `friction` | 0.30 – 1.50 | the biggest sim-to-sim divergence: MuJoCo's elliptic friction cone vs Isaac Gym's pyramidal one |
| `added_mass` | −1.5 – +3.0 kg | payload and battery variation |
| `kp_scale` | 0.80 – 1.20 | motor strength varies with heat and charge |
| `kd_scale` | 0.80 – 1.20 | damping variation |
| `joint_damping` | 0.70 – 1.40 | wear and lubrication |
| `motor_strength` | 0.85 – 1.15 | per-motor manufacturing spread |
| `action_latency` | 0 – 2 steps | 0–40 ms; real robots always have some, simulators default to none |
| `com_offset_x` | ±0.05 m | payload never sits exactly on centre |

Adding a parameter means updating `PARAM_SPECS` **and** `networks.PRIV_DIM` —
a test enforces that they agree.

---

## Compute: read this before starting a real run

This trainer runs on **plain MuJoCo on CPU**, which is why it works on any
machine and why the test suite can exercise it. It is not fast:

| | measured on 8-core CPU |
|---|---|
| Single env | ~1,900 policy steps/s (19,000 physics steps/s) |
| 8 envs, with PPO updates | ~600 policy steps/s |
| 5M steps | **~2.5 hours per 5M… optimistically days for convergence** |

Legged locomotion policies typically need 10⁸–10⁹ environment steps. That is
firmly GPU territory, and this repo's stated constraint is bursty free-tier GPU
access. So:

- **`train.py` as written is correct, runnable and complete** — use it to
  validate the reward function, the curriculum and the export path end to end.
- **For a converged policy, port the env to MJX** (see below). The interface is
  deliberately contained so the swap touches one file.

### Using your local GPU

The trainer picks its device automatically and says so. **NVIDIA, Intel Arc
(XPU), AMD ROCm, and Apple MPS** are all recognised — see
[`docs/DEPENDENCIES.md` §4.1](../docs/DEPENDENCIES.md#41-gpu-flavours--nvidia-intel-arc-amd-apple)
for the install command on each.

```bash
./scripts/setup_env.sh --gpu auto           # matching torch wheel
make hw-profile                             # record this machine
python train.py --smoke                     # both RMA phases, tiny budget
python train.py --run-name v1               # --device auto is the default
python train.py --run-name v1 --device xpu  # insist on Arc; warns if missing
python train.py --tune-profile arc-16gb     # force a sizing profile
```

`config.resolve_device()` never falls back silently — asking for `cuda`/`xpu`
and getting `cpu` without noticing is how you discover three hours later that
the run is 40× slower than expected. When a device is found,
`tune_for_device()` applies a named profile from `tune_profiles.py` (measured
`rtx3050-8gb`; estimated `arc-*` / `cuda-*` / `apple-mps` / `cpu-*` until
someone benchmarks).

A local consumer GPU helps the PPO update, not the simulation — MuJoCo still
steps on CPU here. Measured on an RTX 3050 (8 GB, sm_86):

| | CPU | GPU | speedup |
|---|---|---|---|
| training throughput | ~600 sps | ~1,050 sps | 1.8x |
| 4096³ matmul (the PPO update) | 198 ms | 28 ms | 7.1x |
| 1280x720 offscreen render | 6.6 fps (osmesa) | 309 fps (egl) | 47x |

Note the shape of that: **rendering and raw matmul gain far more than training
does**, because training is still gated by sequential MuJoCo stepping on CPU.
Expect a useful speedup, not a transformative one. The transformative change is
MJX, below, which moves the *physics* onto the GPU.

**Rendering:** `MUJOCO_GL=egl` gives GPU offscreen rendering with no X server,
and is where the GPU pays off most — 47x here, which is the difference between
a video-producing experiment sweep taking minutes and taking an hour. Worth
knowing: EGL returns valid frames at *software* speed when it cannot load the
driver, and raises nothing. The 6.6-vs-309 fps gap above was invisible until
`gpu_check.py` started inspecting stderr for `driver (null)`. Verify with
`make gpu` rather than assuming (NVIDIA path); on Arc/AMD rely on
`check_env.py` + `--smoke`.

Checkpoints land in `stage3-go2-training/runs/<run-name>/` regardless of which
directory you launch from — a relative `out_dir` is anchored to the stage
directory, not the shell's CWD.

### The MJX/GPU path

`env/go2_env.py` is the only backend-specific file. Everything else — the
contract, the networks, the reward terms, the curriculum, the export — is
backend-agnostic.

```bash
pip install mujoco-mjx jax[cuda12]      # not in requirements.txt: GPU-only
```

Porting means: replace the sequential `for env in self.envs` stepping with a
`jax.vmap`'d batched step over the same MJCF, keeping `build_obs`'s field order
identical. `mujoco_playground` provides a reference Go2 env worth reading
first. Everything downstream, including `export.py`, works unchanged.

---

## Configuration

All hyperparameters live in `config.py`. The reward weights are what you will
sweep most:

```python
"tracking_lin_vel": 1.5,    # the objective
"tracking_ang_vel": 0.5,
"alive":            0.5,
"body_height":      5.0,    # strongest stability term
"body_orientation": 1.0,
"lateral_drift":    0.5,    # added because of Exp 1 F1.3
"action_rate":      0.01,   # the key smoothness term for sim-to-real
...
```

**The most common locomotion reward-shaping failure** is a penalty term that
can outweigh `tracking_lin_vel`, producing a policy that stands perfectly
still and collects `alive` forever. If that happens, `info["reward_terms"]`
gives the per-term breakdown — the bug is always visible there and essentially
invisible in the total.

Debugging tip: `--no-curriculum --no-domain-rand` pins the environment so
reward bugs can be isolated. A moving terrain distribution makes them
impossible to see.

---

## Export and evaluation

`export.py` verifies four things before writing any file:

1. eager module shapes satisfy the contract
2. traced graph shapes still satisfy it
3. traced output matches eager output numerically
4. the written files reload and run a full forward pass

Then it writes `body_latest.jit`, `adaptation_module_latest.jit` and
`export_metadata.json` — the exact filenames Stage 2's `paths.py` looks for.

`--random` exports untrained but contract-valid networks. Useful for testing
the pipeline with no training at all, and as a control condition.

`evaluate.py` then runs the export through `harness.run_trial()` and compares
against baseline figures read from the committed Experiment 1 CSV:

```
 cmd_vx | survival |  achieved |  baseline |    delta |  verdict
   0.50 |    100% |     0.009 |     0.227 |   -0.218 |    worse
```

*(that row is a 4,096-step smoke run — it is supposed to be worse)*

---

## Status

| Component | Status |
|---|---|
| Contract definition + guards | ✅ implemented, tested |
| Environment (70-dim obs, CPU MuJoCo) | ✅ implemented, tested |
| Reward terms (12) | ✅ implemented |
| Curriculum from Exp 3 thresholds | ✅ implemented, tested |
| Domain randomisation (8 params) | ✅ implemented, tested |
| PPO + RMA phase 2 | ✅ implemented, smoke-verified |
| Checkpoint / resume | ✅ implemented |
| TorchScript export | ✅ implemented, 4-way verified |
| Evaluation vs baseline | ✅ implemented |
| Gait randomisation (trot/pace/bound per episode) | ✅ implemented, tested |
| Running-speed curriculum level (`flat-run`, up to 2.5 m/s) | ✅ implemented, tested |
| Obstacle-course terrain (`obstacles-easy/hard`) | ✅ implemented, tested — blind, proprioception only |
| **A converged policy across all of the above** | ⏳ **needs GPU compute — see below** |
| MJX/GPU backend | 📋 designed, not ported |

The pipeline is complete and verified end to end. What is missing is compute,
not code.

---

## "All types of movement" — what this can and cannot deliver in one session

Multi-gait, running-speed, and obstacle-course training are wired in above
and pass the test suite, but a single actually-converged multi-skill policy
is a genuinely large training run, not a few-hour job. Being specific about
why, so "not done yet" doesn't read as "not tried":

**Scale.** Production multi-terrain Go2 policies (Isaac Lab / Isaac Gym) train
with **4,096 parallel environments on an RTX 4090**. This repo's env is
plain sequential MuJoCo on CPU, GPU-tuned to **32** parallel envs (see
`config.tune_for_device`) — roughly two orders of magnitude fewer parallel
samples per second. `env/go2_env.py` is deliberately the only backend-specific
file so this is fixable (port to MJX, batched on GPU) without touching the
reward, curriculum, or contract — see "The MJX/GPU path" below — but it is
not ported yet.

**Method.** The strongest published results for natural-looking multi-gait
locomotion (walk, trot, gallop, recovery, all in one policy) use **AMP
(Adversarial Motion Priors)** — training against a discriminator that
compares the policy's motion to a reference motion-capture dataset, rather
than hand-written reward shaping alone. This repo's 12 hand-written reward
terms (`env/rewards.py`) are the same family used by walk-these-ways itself
and are sufficient for functional multi-gait locomotion, but AMP is what
produces gaits an animator would call natural, including a genuine flight
phase for galloping rather than a fast trot.

**What's real right now:** a background training run
(`train.py --run-name multigait_v1 --timesteps 2000000 --device cuda`) is
exercising exactly this — gait sampled per episode from `{trot, pace,
bound}`, `flat-run`'s 2.5 m/s ceiling, and both obstacle levels — on the
RTX 3050. At ~1,070 steps/s that's roughly 30 minutes for 2M steps, nowhere
near convergence (10⁸–10⁹ steps is the typical range for legged locomotion)
but enough to confirm the whole pipeline — new curriculum levels, gait
sampling, obstacle terrain — trains without crashing, curriculum-promotes
correctly, and exports a contract-valid policy at the end.

### References found for extending this further

- **AMP for quadrupeds, open dataset**: [inspirai/MetalHead](https://github.com/inspirai/MetalHead) —
  Unitree A1, includes motion-capture reference clips for
  gallop/jump/trot/turn, trained with AMP. The closest open, license-clear
  starting point for a reference-motion dataset if this repo adds AMP.
- **GPU-batched MuJoCo, the actual target for the MJX port**:
  [MuJoCo Playground](https://playground.mujoco.org/) — JAX/MJX-batched
  environments including Go1 locomotion; the technical report documents the
  exact batching pattern `env/go2_env.py` would need to adopt.
- **Concrete Go2 + MuJoCo Playground + obstacle curriculum reference**:
  [NtagkasAlex/phase_guided_terrain_traversal](https://github.com/NtagkasAlex/phase_guided_terrain_traversal) —
  7-terrain curriculum (flat/wave/slope/rough-slope/stairs-up/stairs-down/
  obstacle) on real Go2 hardware, MuJoCo Playground in sim. Closest published
  analogue to this repo's own curriculum design.
- **Why blind obstacle traversal is the harder problem**: robot parkour work
  ([Robot Parkour Learning](https://arxiv.org/pdf/2309.05665),
  [Humanoid Parkour Learning](https://arxiv.org/pdf/2406.10759)) uses depth
  input specifically because proprioception-only obstacle avoidance (what
  `obstacles-easy/hard/gauntlet` above give you) is a fundamentally harder,
  reactive-only problem — worth reading before assuming vision is optional.

### Motion datasets, for making it move like a real dog

"Walk like a real dog" is what AMP (Adversarial Motion Priors) is *for* — a
discriminator scores the policy's motion against real reference clips, so
gait naturalness becomes part of the reward instead of something hand-tuned
reward terms only approximate. This repo's `env/rewards.py` does not do
this; it's twelve hand-written terms, the same family walk-these-ways itself
used. Adding AMP is a real project, not a config flag: a discriminator
network, a motion-matching observation window, retargeting whichever dataset
below onto the Go2's joint layout, and a reward term that blends the
discriminator score with the existing tracking terms. Datasets found, in
order of how directly they answer "real dog":

- **[Tencent-RoboticsX/lifelike-agility-and-play](https://github.com/Tencent-RoboticsX/lifelike-agility-and-play)** —
  actual Labrador retriever motion capture (`.bvh`), plus the same clips
  already retargeted to a quadruped robot's joint layout (`data/mocap_data`).
  This is the one directly labeled "real dog." **License is `NOASSERTION`
  on GitHub** — check before any commercial use, and re-verify current terms
  before integrating; not resolved by this session.
- **[inspirai/MetalHead](https://github.com/inspirai/MetalHead)** — not a
  real dog, but real AMP-ready reference clips for a robot (Unitree A1):
  gallop_forward, jump, trot_forward, turn. Lower retargeting effort than
  the Labrador data since it's already robot-joint data, at the cost of
  being animal-*inspired* rather than animal-*recorded*.
- **[Kine2Go](https://arxiv.org/abs/2606.14433)** (arXiv, 2026-06) — 800+
  Go2-specific kinematic trajectories from 40 distinct trained policies, not
  animal mocap at all. Useful as a "what does good Go2 locomotion look like"
  reference set rather than a naturalness target; no confirmed public
  download link found, only the paper.
- **Truebones Zoo** — 1,038 clips across 70 skeletons including quadrupeds,
  frequently cited in animal-mocap literature. Distributed as a paid Unity
  asset in the versions found; treat as license-restricted, not a free
  drop-in.

None of these are wired into this repo. Retargeting mocap onto the Go2's
specific 12-DOF layout and alternating hip signs (see `CLAUDE.md`'s hard
invariants) is itself nontrivial work, independent of which dataset is
chosen.

---

## Related

- [`docs/TDD.md` §9](../docs/TDD.md) — the design this implements
- [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) — the contract in full
- [`../stage2-go2-mujoco-inference/experiments/results/EXPERIMENT_FINDINGS.md`](../stage2-go2-mujoco-inference/experiments/results/EXPERIMENT_FINDINGS.md) — the measurements that shaped this
- [`../reinforcement-learning-theories/chapter-07-legged-locomotion-with-rl.md`](../reinforcement-learning-theories/chapter-07-legged-locomotion-with-rl.md) — RMA and gait conditioning theory
