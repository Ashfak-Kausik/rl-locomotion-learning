# Feature List

Every capability in this repository, its status, where it lives, and how it was
verified. Derived by reading the code, not the roadmap.

**Legend**

| | Meaning |
|---|---|
| ✅ | Implemented and verified |
| ⚠️ | Implemented with a caveat |
| 🔄 | Declared in progress; **no code in the tree** |
| 📋 | Planned, specified, not started |
| ❌ | Missing, and its absence matters |

---

## Summary

| Area | ✅ | ⚠️ | 🔄 | 📋 | ❌ |
|---|---|---|---|---|---|
| Stage 1 — RL fundamentals | 12 | 0 | 0 | 0 | 1 |
| Stage 2 — Go2 inference | 24 | 2 | 0 | 0 | 0 |
| Research / experiments | 17 | 1 | 0 | 2 | 0 |
| Documentation | 12 | 1 | 0 | 0 | 0 |
| Infrastructure | 14 | 0 | 0 | 0 | 4 |
| Stage 3 — training | 0 | 0 | 8 | 0 | 0 |
| Stages 4–6 | 0 | 0 | 0 | 14 | 0 |

**Headline:** everything the project claims to have *delivered* is genuinely
delivered. The gap is Stage 3 (declared "in progress", zero code) and testing
infrastructure (absent entirely).

---

## 1. Stage 1 — RL fundamentals ✅

| Feature | Status | Where | Verified by |
|---|---|---|---|
| PPO on CartPole-v1 | ✅ | `01_cartpole_ppo.py` | mean reward ~500 / 500 |
| PPO on LunarLander-v3 | ✅ | `03_lunarlander_ppo.py` | 220–274 (solved ≥ 200) |
| PPO on Pendulum-v1 | ✅ | `05_pendulum_ppo.py` | ≥ −200 at 300k steps |
| Discrete action spaces | ✅ | envs 1–2 | — |
| Continuous action spaces | ✅ | Pendulum, gSDE | `train/std` present |
| Vectorised environments | ✅ | `DummyVecEnv`, 8 envs | envs 2–3 |
| Episode monitoring | ✅ | `VecMonitor` | TensorBoard rollout metrics |
| Model persistence | ✅ | `model.save/load` | `.zip` round-trip |
| Deterministic evaluation | ✅ | `evaluate_policy` | 10–100 episodes |
| Rendered playback | ✅ | `02/04/06_*_watch.py` | `render_mode="human"` |
| TensorBoard logging | ✅ | `tb_logs/<env>/PPO_N` | multi-run overlay |
| Hyperparameter comparison | ✅ | 4 runs per env | documented in README |
| Best-checkpoint callback | ❌ | — | `EvalCallback` never used, despite the README identifying policy collapse as the reason you need it |

**Documented learning outcomes** (`stage1-rl-fundamentals/README.md`, 179
lines): policy collapse at ~860k steps on LunarLander; `clip_fraction` 0.45 and
`approx_kl` 0.08 on Pendulum diagnosing an over-aggressive learning rate;
`ep_len_mean` being uninformative for Pendulum by construction.

---

## 2. Stage 2 — Go2 inference ✅

### 2.1 Model handling

| Feature | Status | Where |
|---|---|---|
| Load Go2 MJCF | ✅ | `01_hello_go2.py` |
| Self-contained model (no external checkout) | ✅ | `scenes/go2_model/` — 16 meshes |
| Model introspection (bodies/joints/actuators/sensors) | ✅ | `02_inspect_go2.py` |
| Keyframe reset + training-pose override | ✅ | all runtime scripts |
| Portable path resolution | ✅ | `paths.py` |
| Scene selection by CLI argument | ✅ | `08_view_scene.py` |

### 2.2 Observation construction

| Feature | Status | Where |
|---|---|---|
| 70-dim observation vector | ✅ | `harness.py::build_obs` |
| Projected gravity (body frame) | ✅ | `quat_rotate_inverse` |
| 15-dim scaled command vector | ✅ | `COMMANDS_SCALE` |
| Joint positions relative to default pose | ✅ | — |
| Scaled joint velocities | ✅ | ×0.05 |
| Two-step action history in the observation | ✅ | `last_action`, `prev_action` |
| 4 per-foot clock signals | ✅ | `sin(2π·phase)` |
| 30-frame rolling history | ✅ | `deque(maxlen=30)` |
| Shape assertion | ✅ | `assert obs.shape == (70,)` |
| Superseded obs implementation retained | ⚠️ | `04_build_obs_vector.py` — wrong layout **and** wrong hip signs, no warning in the file |

### 2.3 Policy inference

| Feature | Status | Where |
|---|---|---|
| TorchScript checkpoint loading | ✅ | `torch.jit.load` |
| RMA adaptation module (2100 → 2) | ✅ | — |
| Body network (2102 → 12) | ✅ | — |
| Gradient-free inference | ✅ | `torch.no_grad()` |
| Network shape verification tool | ✅ | `05_inspect_policy.py` |
| Actionable missing-weights error | ✅ | `paths.py::require_policy` |
| Loaded-policy shape validation | ⚠️ | not implemented — a wrong checkpoint fails with a raw torch error |

### 2.4 Control

| Feature | Status | Where |
|---|---|---|
| Action → joint target conversion | ✅ | scale 0.25 |
| Hip action-scale reduction | ✅ | ×0.5 at indices 0,3,6,9 |
| PD torque controller | ✅ | Kp 25, Kd 0.6 |
| Dual-rate loop (50 Hz / 500 Hz) | ✅ | `DECIMATION = 10` |
| Gait clock on the policy clock | ✅ | — |
| Manual PD pose holding | ✅ | `03_pose_go2.py`, Kp 100 / Kd 2 |

### 2.5 Locomotion capability

| Feature | Status | Evidence |
|---|---|---|
| Sustained forward walking | ✅ | Exp 1: 100% survival, all speeds |
| Velocity command tracking | ⚠️ | works, but achieves only 37–55% of command — **this is the research finding**, not a defect |
| Trot gait | ✅ | Exp 2: 100% survival |
| Pace gait | ✅ | Exp 2: 100% survival, **best tracking** |
| Bound gait | ✅ | Exp 2: 100% survival, most rigid posture |
| Yaw / strafe commands | ✅ | `07`, not swept experimentally |
| Slope traversal | ✅ | up to 10° |
| Stair traversal | ⚠️ | 2 cm only; ≥ 5 cm safe-stalls |
| Posture stability | ✅ | height std 0.006–0.007 m at every speed |

### 2.6 Interaction

| Feature | Status | Where |
|---|---|---|
| Interactive viewer | ✅ | `launch_passive` |
| Keyboard velocity control | ✅ | `07` — arrow keys |
| Keyboard strafe | ✅ | `,` / `.` |
| Runtime gait switching | ✅ | keys 7/8/9 |
| Emergency stop (zero commands) | ✅ | key 0 |
| Runtime reset | ✅ | key R |
| Live telemetry printout | ✅ | 1 Hz |
| Offscreen screenshot capture | ✅ | `08` + imageio |

> Letter keys W/A/S/D are reserved by MuJoCo's viewer for rendering shortcuts,
> which is why this repo uses arrow keys. The docstring in `07` documents a
> W/A/S/D mapping that the code does not implement — read the code, not the
> docstring header.

---

## 3. Research / experiments ✅

### 3.1 Harness

| Feature | Status |
|---|---|
| Headless trial execution | ✅ |
| Parameterised by scene / velocity / gait / seed | ✅ |
| Seeded initial-condition randomisation | ✅ |
| Settle window before measurement | ✅ 3 s |
| Fall detection (height + tilt) | ✅ 0.15 m / 60° |
| Fall-time recording | ✅ 2 ms resolution |
| Network injection (load once, reuse) | ✅ |
| Metrics dict return, no I/O | ✅ |
| Built-in smoke test | ✅ `__main__` |
| CLI configuration of sweeps | 📋 constants only |
| Run provenance in results | 📋 no git SHA / versions |

### 3.2 Metrics

| Metric | Status |
|---|---|
| Velocity tracking error | ✅ |
| Lateral drift | ✅ |
| Body height mean / std | ✅ |
| Distance travelled | ✅ |
| Survival rate | ✅ |
| Time to fall | ✅ |
| Traversability (survival **and** progress) | ✅ — corrected mid-study |
| Energy / cost of transport | 📋 not measured |

### 3.3 Terrain

| Feature | Status |
|---|---|
| Parametric slope generation (5–25°) | ✅ |
| Parametric stair generation (2–16 cm) | ✅ |
| Flat run-up in every scene | ✅ 2.5 m |
| Byte-identical regeneration | ✅ verified |
| Visual terrain inspection | ✅ `check_terrain_visual.py` |
| Heightfield / rough terrain | 📋 Stage 4 |

### 3.4 Studies

| Study | Status | Scale | Headline finding |
|---|---|---|---|
| Exp 1 — velocity sweep | ✅ | 6 × 5 | policy achieves 37–55% of commanded velocity; saturates ~0.55 m/s |
| Exp 2 — gait robustness | ✅ | 3 × 5 | **pace transfers better than trot** at 0.5 m/s |
| Exp 3 — terrain robustness | ✅ | 10 × 5 | 10° slopes OK, 5 cm steps not — continuous vs discrete perturbations transfer differently |
| Gait × velocity grid | 📋 | 0 | flagged as a limitation in the findings; ~90 trials against existing code |

### 3.5 Outputs

| Feature | Status |
|---|---|
| Per-trial CSV | ✅ 3 files, 75 rows |
| Printed summary tables | ✅ mean ± std over survivors |
| Interpreted findings document | ✅ `EXPERIMENT_FINDINGS.md` |
| Data figures | ✅ 4 PNG @ 300 dpi |
| Qualitative figures | ✅ 7 curated panels |
| Figures regenerable without policy | ✅ verified |
| Run logs | ✅ committed `.txt` |

---

## 4. Documentation ✅

| Document | Status | Size |
|---|---|---|
| RL theory chapters 1–8 | ✅ | ~1,090 lines |
| Stage 1 metric analysis | ✅ | 179 lines |
| Stage 2 debugging insights | ✅ | 22 lines |
| Experiment findings | ✅ | detailed, per-experiment |
| Root README | ⚠️ | truncated at "Repository Structure" |
| `ARCHITECTURE.md` | ✅ | text diagrams |
| `TECH-STACK-PRIMER.md` | ✅ | tech basics |
| `DEPENDENCIES.md` | ✅ | checklist + troubleshooting |
| `DOCKER.md` | ✅ | container workflows |
| `SRS.md` | ✅ | requirements |
| `TDD.md` | ✅ | design decisions |
| `FEATURES.md` | ✅ | this file |
| `REVERSE-ENGINEERING.md` | ✅ | findings + first tasks |
| `AGENTIC-WORKFLOW.md` | ✅ | AI-assisted development |
| `CLAUDE.md` | ✅ | agent context file |

---

## 5. Infrastructure

| Feature | Status | Where |
|---|---|---|
| Dependency manifest | ✅ | `requirements.txt` |
| Cross-distro auto-installer | ✅ | `scripts/setup_env.sh` |
| 4-layer environment verifier | ✅ | `scripts/check_env.py` |
| Machine-independent paths | ✅ | `paths.py` |
| Docker image (CPU-only) | ✅ | `docker/Dockerfile` |
| Compose: interactive shell | ✅ | `lab` |
| Compose: headless batch | ✅ | `headless` |
| Compose: X11 GUI | ✅ | `viewer` |
| Compose: TensorBoard | ✅ | `tensorboard` |
| Host-uid file ownership | ✅ | build args |
| Headless rendering (OSMesa) | ✅ | verified in container |
| Make targets | ✅ | `Makefile` |
| `.gitignore` for large artefacts | ✅ | rewritten |
| `.dockerignore` | ✅ | — |
| **Automated tests** | ❌ | no `tests/`, no pytest |
| **CI pipeline** | ❌ | no workflow |
| **Linter / formatter config** | ❌ | none |
| **Dependency lockfile** | ❌ | ranges only, no hashes |

---

## 6. Stage 3 — custom training 🔄

Declared "in progress" in the README. **No code exists in the tree.** Nothing
imports MJX or `mujoco_playground`.

| Feature | Status |
|---|---|
| MJX / `mujoco_playground` Go2 environment | 🔄 |
| Locomotion reward function | 🔄 |
| Domain randomisation | 🔄 |
| Terrain curriculum | 🔄 |
| PPO training loop for free-tier GPU | 🔄 |
| Checkpoint / resume across sessions | 🔄 |
| TorchScript export in the existing contract | 🔄 |
| Evaluation via the existing harness | 🔄 |

Design sketch in [TDD.md §9](TDD.md#9-design-for-stage-3). The critical
requirement is contract compatibility — honour the 70/2100/2102/12 interface
and every Stage 2 measurement tool works on the new policy unchanged.

---

## 7. Stages 4–6 📋

### Stage 4 — vision-conditioned locomotion

| Feature | Status |
|---|---|
| Depth-camera sensors in MJCF | 📋 |
| Extended observation space | 📋 |
| CNN / latent depth encoder | 📋 |
| Heightfield terrain | 📋 |
| Traverse terrain the blind policy fails | 📋 |

Directly motivated by Exp 3 finding F3.3: blind proprioception handles
continuous slopes but not discrete steps.

### Stage 5 — ROS2 deployment

| Feature | Status |
|---|---|
| ROS2 inference node | 📋 |
| Velocity-command subscriber | 📋 |
| Joint-command publisher | 📋 |
| Unitree SDK2 integration | 📋 |
| Real-time guarantees | 📋 |
| Emergency stop | 📋 |

### Stage 6 — sim-to-real

| Feature | Status |
|---|---|
| Hardware deployment | 📋 |
| Sim-to-real gap measurement | 📋 |
| Failure-mode documentation | 📋 |
| Domain-randomisation iteration | 📋 |

---

## 8. The four things worth doing first

Ranked by value ÷ risk. Full reasoning in
[REVERSE-ENGINEERING.md §9](REVERSE-ENGINEERING.md#9-suggested-first-contributions).

1. **Test suite** ❌ — six tests, none needing policy weights, guarding the
   observation contract and model dimensions. Nothing else is safe to refactor
   until this exists.
2. **CI workflow** ❌ — run `check_env.py` plus those tests on every push.
3. **Gait × velocity grid** 📋 — 90 trials against code that already exists.
   Experiment 2's surprising pace-beats-trot result is currently single-speed,
   and the findings document flags that limitation itself. This is a real
   research result reachable on a CPU tonight.
4. **`EvalCallback` in Stage 1** ❌ — the README diagnoses policy collapse and
   explains why best-checkpoint saving matters, then never uses it.

---

## Related

- [REVERSE-ENGINEERING.md](REVERSE-ENGINEERING.md) — how these statuses were determined
- [SRS.md](SRS.md) — requirements behind each feature
- [TDD.md](TDD.md) — why each was built this way
- [ARCHITECTURE.md](ARCHITECTURE.md) — how they connect
