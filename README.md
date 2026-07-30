# Quadruped Locomotion via Reinforcement Learning

End-to-end research project on reinforcement learning for legged robot locomotion, with a focus on the Unitree Go2 quadruped platform. Covers RL algorithm fundamentals, simulation infrastructure, policy deployment, and a planned trajectory toward vision-conditioned and hardware-deployable controllers.

> **Current milestone:** Pretrained `walk-these-ways` policy successfully deployed on the Unitree Go2 in MuJoCo. Robot achieves sustained forward locomotion at ~0.28 m/s.
> **Active work:** Custom policy training on free-tier GPU infrastructure.
> **Platform constraint:** Developed on CPU-only hardware; GPU usage limited to bursty training sessions on cloud platforms.

---
![Go2 Walking in MuJoCo](media/stage2_walking.gif)
---

## Project Scope

Reinforcement learning has matured into the dominant approach for legged-robot locomotion control, with frameworks like `walk-these-ways` (MIT Improbable AI Lab) demonstrating that gait-conditioned policies trained entirely in simulation can transfer to physical hardware. This project rebuilds, deploys, and extends that body of work for the Unitree Go2 platform, with three primary objectives:

1. **Establish a reproducible CPU-friendly deployment pipeline** for pretrained legged-robot policies in MuJoCo, suitable for development environments without dedicated GPU hardware.
2. **Train custom locomotion policies** under bursty-GPU constraints using modern parallel simulators (MJX, `mujoco_playground`).
3. **Extend toward vision-conditioned locomotion** and hardware deployment for terrain-aware, autonomous operation.

The methodology emphasizes principled engineering — reading source code over relying on documentation, verifying assumptions against training configurations, and isolating sim-to-sim transfer issues systematically.

---

## Current Results

### Pretrained Policy Deployment (MuJoCo)

The `walk-these-ways` policy — originally trained in Isaac Gym on a GPU cluster — has been deployed on the Unitree Go2 MJCF (from `mujoco_menagerie`) running entirely on CPU. Key results:

| Metric | Value |
|---|---|
| Sustained forward velocity | ~0.28 m/s (commanded 0.5 m/s) |
| Body height stability | 0.275 m ± 0.005 m |
| Continuous walking duration | 60+ seconds without failure |
| Control frequency | Policy 50 Hz, simulation 500 Hz, decimation 10 |
| Inference latency | <2 ms per policy step (CPU) |

### Sim-to-Sim Transfer Findings

Migrating a policy trained in Isaac Gym to MuJoCo surfaced several non-trivial mismatches. The principled fixes from this work are documented in [`stage2-go2-mujoco-inference/`](./stage2-go2-mujoco-inference/) and summarized below:

| Issue | Resolution |
|---|---|
| Control frequency mismatch | Decimation tuned to match the policy's 50 Hz training frequency |
| Gait clock signal misinterpretation | Per-foot phase sines reconstructed from training source code |
| MJCF default pose ≠ training default pose | Joint initialization overridden to match training configuration |
| Hip sign convention inversion | Joint angle signs corrected against training config dump |

---

## Project Components

### Component 1 — RL Algorithm Foundation

Implementation and analysis of Proximal Policy Optimization (PPO) on standard Gymnasium benchmarks. Establishes the algorithmic baseline used throughout subsequent work and a working understanding of training dynamics, hyperparameter sensitivity, and convergence diagnostics.

📁 [`stage1-rl-fundamentals/`](./stage1-rl-fundamentals/)

**CartPole-v1 — discrete control baseline**

> 4-dim observation, 2-dim discrete action, dense reward. Establishes the train-evaluate loop and policy/value network roles.

| Result | 500/500 (max) |
|--|--|
| Timesteps | 25,000 |
| Compute | ~2 min, CPU |

![CartPole](media/stage1_cartpole.gif)

**LunarLander-v3 — shaped-reward control**

> 8-dim observation, 4-dim discrete action, multi-component shaped reward. Introduces reward-engineering tradeoffs.

| Result | 246 (solved threshold 200) |
|--|--|
| Timesteps | 300,000 |
| Compute | ~15 min, CPU |

![LunarLander](media/stage1_lunarlander.gif)

**Pendulum-v1 — continuous control**

> 3-dim observation, 1-dim continuous action, Gaussian policy. Continuous action spaces are the algorithmic prerequisite for joint-level robot control.

| Result | −151 (solved threshold −200) |
|--|--|
| Timesteps | 400,000 |
| Compute | ~10 min, CPU |

![Pendulum](media/stage1_pendulum.gif)

---

### Component 2 — Go2 Policy Deployment in MuJoCo

Full inference pipeline from MuJoCo state → 70-dim observation vector → adaptation module → policy network → joint targets → PD-controlled torques. Implements the architecture of Rapid Motor Adaptation (RMA) deployment as used by `walk-these-ways`.

📁 [`stage2-go2-mujoco-inference/`](./stage2-go2-mujoco-inference/)

| Sub-task | Outcome |
|---|---|
| **2.1** — Load Go2 MJCF, verify viewer | Robot loads and renders correctly |
| **2.2** — Inspect model structure | 19 qpos, 18 qvel, 12 actuators enumerated |
| **2.3** — Manual PD pose holding | Stable standing pose under gravity |
| **2.4** — Build 70-dim observation vector | Each field traced to training source code |
| **2.5** — Load policy, run inference | Robot walks (see Current Results above) |

---

## Tech Stack

| Tool | Version | Role |
|---|---|---|
| Python | 3.10 | Implementation |
| Stable-Baselines3 | 2.x | PPO baseline for RL fundamentals |
| Gymnasium | 0.29+ | Standard RL environments |
| MuJoCo | 3.x | Physics simulation |
| PyTorch | 2.1+ | Policy inference and (later) training |
| TensorBoard | latest | Training diagnostics |
| MJX / mujoco_playground | TBD | Planned for GPU-accelerated training |

---

## Stage Planning & Status

| Stage | Plan | Description | Status |
|---|---|---|---|
| **Stage 1** | RL fundamentals | Implement PPO on CartPole, LunarLander, and Pendulum to establish the algorithmic baseline. Build training-curve diagnostics intuition before applying RL to robotic control. | ✅ Complete |
| **Stage 2** | Go2 inference in MuJoCo | Deploy the pretrained `walk-these-ways` policy on the Unitree Go2 in MuJoCo. Implement the full inference pipeline: observation construction, history buffer, adaptation module, dual-rate PD control. | ✅ Complete |
| **Stage 3** | Custom Go2 policy training | Train a locomotion policy from scratch using MJX or `mujoco_playground` on free-tier GPU (Colab / Kaggle / Lightning AI). Investigate reward shaping, curriculum design, and domain randomization for transferability. | 🔄 In Progress |
| **Stage 4** | Vision-conditioned locomotion | Integrate depth-camera input into the policy's observation space for terrain-aware locomotion (stairs, gaps, slopes). Extends from blind proprioceptive control to perceptive control. | ⏳ Planned |
| **Stage 5** | ROS2 deployment infrastructure | Wrap the trained policy as a ROS2 node compatible with Unitree SDK2 deployment requirements. Establish the software interface required for hardware testing. | ⏳ Planned |
| **Stage 6** | Sim-to-real validation | Quantify the sim-to-real gap on the physical Go2 platform (subject to hardware availability). Document failure modes and iterate on domain randomization parameters. | 🔮 Future |

---

## Repository Structure

```
rl-locomotion-learning/
├── reinforcement-learning-theories/   8 chapters: RL foundations → PPO →
│                                      legged locomotion → this project
├── stage1-rl-fundamentals/            PPO on CartPole / LunarLander / Pendulum
│   └── 01-06_*.py                     odd = train + save, even = load + watch
│
├── stage2-go2-mujoco-inference/       The core of the project
│   ├── paths.py                       env-overridable path resolution
│   ├── 01-08_*.py                     numbered curriculum; 06 = the robot walks
│   ├── scenes/                        self-contained Go2 MJCF + 11 terrain worlds
│   ├── experiments/                   harness + 3 studies + figure pipeline
│   │   ├── harness.py                 headless run_trial() — the shared backbone
│   │   ├── exp1/exp2/exp3_*.py        velocity, gait, terrain sweeps
│   │   └── results/                   CSVs + EXPERIMENT_FINDINGS.md
│   └── paper_figures/                 curated figures for the write-up
│
├── docs/                              architecture, SRS, TDD, features, setup
├── scripts/                           setup_env.sh, check_env.py
├── docker/                            Dockerfile + 4 compose services
├── policies/                          walk-these-ways checkpoints (not committed)
└── media/                             demo GIFs
```

---

## Getting Started

```bash
git clone <repo> && cd rl-locomotion-learning
./scripts/setup_env.sh          # detects your OS, installs everything, verifies
source .venv/bin/activate
make help                       # see every available target
```

Or run it containerised, with no host dependencies beyond Docker:

```bash
docker compose -f docker/compose.yaml build
docker compose -f docker/compose.yaml run --rm lab
```

> **Note on the policy weights.** The pretrained `walk-these-ways` checkpoints
> are a large external *input* to this project and are not committed. Stage 2
> inference needs them; see [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md#6-the-one-dependency-we-cannot-install).
> Everything else — all of Stage 1, Stage 2 scripts 01–04, scene generation, and
> the complete figure pipeline — runs without them.

### Documentation

| Document | Read when |
|---|---|
| [docs/README.md](docs/README.md) | starting out — includes a day-one reading order |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | you want the system view, with diagrams |
| [docs/TECH-STACK-PRIMER.md](docs/TECH-STACK-PRIMER.md) | MuJoCo / PyTorch / Gymnasium are new to you |
| [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md) | setting up, or something will not install |
| [docs/DOCKER.md](docs/DOCKER.md) | running in a container |
| [docs/SRS.md](docs/SRS.md) · [docs/TDD.md](docs/TDD.md) | requirements and design rationale |
| [docs/FEATURES.md](docs/FEATURES.md) | what is implemented vs planned |
| [docs/REVERSE-ENGINEERING.md](docs/REVERSE-ENGINEERING.md) | known issues and good first tasks |
| [docs/AGENTIC-WORKFLOW.md](docs/AGENTIC-WORKFLOW.md) | working on this repo with AI coding tools |
