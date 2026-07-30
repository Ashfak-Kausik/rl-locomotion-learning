# Documentation

Onboarding and reference material for **rl-locomotion-learning** — a research
project that deploys a quadruped locomotion policy trained in NVIDIA Isaac Gym
into MuJoCo on CPU, and measures what the transfer costs.

---

## Start here

**If you have never touched RL or robotics simulation**, read in this order.
Roughly one focused day.

| # | Read | Time | You will understand |
|---|---|---|---|
| 1 | [ARCHITECTURE.md §1–2](ARCHITECTURE.md) | 10 min | what this project *is* |
| 2 | [DEPENDENCIES.md §1](DEPENDENCIES.md) → run `./scripts/setup_env.sh` | 15 min | a working environment |
| 3 | [TECH-STACK-PRIMER.md](TECH-STACK-PRIMER.md) | 60 min | Python/NumPy/MuJoCo/PyTorch/Gymnasium as used here |
| 4 | `../reinforcement-learning-theories/` ch. 1–5 | 2 h | RL from MDPs through PPO |
| 5 | Run Stage 1: `01_cartpole_ppo.py` → `02_cartpole_watch.py` | 30 min | the train→save→watch loop, first-hand |
| 6 | `../reinforcement-learning-theories/` ch. 6–7 | 1 h | continuous control and legged locomotion |
| 7 | [ARCHITECTURE.md](ARCHITECTURE.md) in full | 45 min | the inference pipeline in detail |
| 8 | Read `01`–`06` in `stage2-go2-mujoco-inference/`, in order | 90 min | how the pipeline is built, one concept at a time |
| 9 | [REVERSE-ENGINEERING.md](REVERSE-ENGINEERING.md) | 30 min | what is solid, what is broken, what to do first |
| 10 | [AGENTIC-WORKFLOW.md](AGENTIC-WORKFLOW.md) | 30 min | how to use AI tools here without breaking things |

**If you already know RL:** [ARCHITECTURE.md](ARCHITECTURE.md) →
`experiments/harness.py` → `results/EXPERIMENT_FINDINGS.md` →
[REVERSE-ENGINEERING.md](REVERSE-ENGINEERING.md). About an hour.

**If you just need it running:** `./scripts/setup_env.sh`, then
[DEPENDENCIES.md §7](DEPENDENCIES.md#7-what-runs-without-it) for what works
without the policy weights.

---

## The documents

### Understanding the system

| Document | Contents |
|---|---|
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | System context, repo map, the full inference pipeline, the dual-rate control loop, the 70-dim observation layout, state/index conventions, the experiment subsystem — all with text diagrams. **The main document.** |
| **[TECH-STACK-PRIMER.md](TECH-STACK-PRIMER.md)** | Python, NumPy, MuJoCo, MJCF, PyTorch/TorchScript, Gymnasium, Stable-Baselines3, TensorBoard, Matplotlib — taught through code from this repo. Ends with a gotcha table. |
| **[FEATURES.md](FEATURES.md)** | Every capability, its status (✅ ⚠️ 🔄 📋 ❌), location and verification. |

### Building and running

| Document | Contents |
|---|---|
| **[DEPENDENCIES.md](DEPENDENCIES.md)** | Complete checklist, three install paths, what runs without the policy weights, troubleshooting table. |
| **[DOCKER.md](DOCKER.md)** | Four compose services, volume layout, rendering backends, file ownership, troubleshooting. |

### Engineering documents

| Document | Contents |
|---|---|
| **[SRS.md](SRS.md)** | Functional and non-functional requirements, reverse-engineered, each marked implemented or not. Includes the data schema and a verification matrix. |
| **[TDD.md](TDD.md)** | Ten design decisions with rejected alternatives, component design, error-handling philosophy, extension points, known debt, and a design sketch for Stage 3. |
| **[REVERSE-ENGINEERING.md](REVERSE-ENGINEERING.md)** | What is actually implemented, 13 findings, claim verification, and a ranked list of first contributions. |

### Working on it

| Document | Contents |
|---|---|
| **[AGENTIC-WORKFLOW.md](AGENTIC-WORKFLOW.md)** | Context files, skills, subagents, hooks, MCP, Cursor rules, task patterns, verification discipline, nine ways agents fail on *this* codebase, research skills, OSS references. |
| **[`../CLAUDE.md`](../CLAUDE.md)** | The context file itself — hard invariants, conventions, known issues. |

### Research

| Document | Contents |
|---|---|
| `../reinforcement-learning-theories/` | 8 chapters (~1,090 lines): RL foundations → PPO → legged locomotion → this project in detail |
| `../stage1-rl-fundamentals/README.md` | Metric-by-metric analysis of every Stage 1 run |
| `../stage2-go2-mujoco-inference/README.md` | The sim-to-sim debugging insights |
| `../stage2-go2-mujoco-inference/experiments/results/EXPERIMENT_FINDINGS.md` | Numbered findings from all three studies |

---

## Quick reference

### Commands

```bash
./scripts/setup_env.sh              # install everything, verify
source .venv/bin/activate
python scripts/check_env.py         # 4-layer environment report
make help                           # all convenience targets
```

```bash
# Runs with NO policy weights
python stage1-rl-fundamentals/01_cartpole_ppo.py
python stage2-go2-mujoco-inference/01_hello_go2.py
python stage2-go2-mujoco-inference/experiments/make_figures.py
python stage2-go2-mujoco-inference/experiments/generate_terrain_scenes.py

# Needs policy weights (see DEPENDENCIES.md §6)
python stage2-go2-mujoco-inference/06_run_policy.py
python stage2-go2-mujoco-inference/07_run_policy_interactive.py
python stage2-go2-mujoco-inference/experiments/exp1_velocity_sweep.py
```

```bash
# Docker
docker compose -f docker/compose.yaml build
docker compose -f docker/compose.yaml run --rm lab
docker compose -f docker/compose.yaml run --rm headless <command>
```

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `GO2_POLICY_DIR` | `<repo>/policies/walk-these-ways-go2` | where the `.jit` checkpoints live |
| `GO2_SCENE` | `go2_flat.xml` | scene filename inside `scenes/` |
| `GO2_MODEL_PATH` | derived from `GO2_SCENE` | full path to a scene XML |
| `MUJOCO_GL` | `glfw` | `osmesa` or `egl` for headless rendering |

### The numbers that organise everything

```
  70    observation dimensions        30    frames of history
  2100  flattened history (30 × 70)   2     adaptation-module latent dims
  2102  body network input            12    action dimensions (joint deltas)
  19    qpos          18  qvel        12    actuators
  500   Hz physics    50  Hz policy   10    decimation
  0.25  action scale  25 / 0.6        Kp / Kd
```

---

## Project status

| Stage | Status |
|---|---|
| 1 — RL fundamentals | ✅ complete, all 3 environments solved |
| 2 — Go2 inference in MuJoCo | ✅ complete, robot walks |
| Research layer | ✅ 3 experiments, 75 trials, findings documented |
| Infrastructure | ✅ deps, Docker, portable paths, docs |
| 3 — Custom policy training | 🔄 declared in progress, **no code in tree** |
| 4 — Vision-conditioned locomotion | 📋 planned |
| 5 — ROS2 deployment | 📋 planned |
| 6 — Sim-to-real | 🔮 hardware-dependent |

**Biggest engineering gap:** no automated tests. See
[REVERSE-ENGINEERING.md §9](REVERSE-ENGINEERING.md#9-suggested-first-contributions).
