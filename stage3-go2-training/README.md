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

| Level | Scene | Baseline result (Exp 3) |
|---|---|---|
| 0 `flat-slow` | flat | trivial |
| 1 `flat-fast` | flat | full command range |
| 2 `slope-10` | 10° | 100% survival, traversable |
| **3 `slope-15`** | 15° | **20% survival — baseline breaks here** |
| 4 `stairs-5` | 5 cm | **safe stall**: 0.47 m, never climbs |
| 5 `slope-20` | 20° | 0% survival |
| 6 `stairs-8` | 8 cm | far beyond baseline |

Levels 0–2 reproduce the baseline's competence. **Level 3 onward is where a new
policy has to actually beat it** — `BASELINE_CEILING = 3` marks that line, so
progress is measurable against a real number rather than a vibe.

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

The trainer picks its device automatically and says so:

```bash
make gpu                                    # is the GPU usable at all?
python train.py --run-name v1               # --device auto is the default
python train.py --run-name v1 --device cuda # insist, and warn loudly if not
```

`config.resolve_device()` never falls back silently — asking for `cuda` and
getting `cpu` without noticing is how you discover three hours later that the
run is 40x slower than expected. When a GPU is found, `tune_for_device()`
rescales the PPO update, because the bottleneck moves:

| | CPU | GPU |
|---|---|---|
| bottleneck | MuJoCo stepping | kernel-launch overhead |
| `num_envs` | 8 | 32 |
| `minibatch_size` | 128 | 2048 |
| TF32 matmuls | n/a | enabled (Ampere+) |

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
`make gpu` rather than assuming.

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
| **A converged policy** | ⏳ **needs GPU compute — see above** |
| MJX/GPU backend | 📋 designed, not ported |

The pipeline is complete and verified end to end. What is missing is compute,
not code.

---

## Related

- [`docs/TDD.md` §9](../docs/TDD.md) — the design this implements
- [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) — the contract in full
- [`../stage2-go2-mujoco-inference/experiments/results/EXPERIMENT_FINDINGS.md`](../stage2-go2-mujoco-inference/experiments/results/EXPERIMENT_FINDINGS.md) — the measurements that shaped this
- [`../reinforcement-learning-theories/chapter-07-legged-locomotion-with-rl.md`](../reinforcement-learning-theories/chapter-07-legged-locomotion-with-rl.md) — RMA and gait conditioning theory
