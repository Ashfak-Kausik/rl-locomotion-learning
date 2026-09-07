# Training Configuration — Pretrained Policy

This document records how the checkpoint deployed by this repository
(`body_latest.jit` + `adaptation_module_latest.jit`) was actually trained.
None of this configuration lives in `rl-locomotion-learning` itself — the
policy is trained externally and used here strictly at inference time (see
`AUDIT.md` Part A.7: zero-shot, no gradients, no fine-tuning anywhere in this
repo). This file exists so a reviewer does not have to take "trained
externally in a GPU-based simulator" on faith.

Everything below is sourced from the external training run's own saved
config (`parameters_cpu.pkl`, an Isaac Gym `Cfg` object) and `outputs.log`,
read read-only from the sibling repository that produced this checkpoint
(see **Provenance chain** below). It is not derivable from this repo alone.

## Provenance chain

1. **[Improbable-AI/walk-these-ways](https://github.com/Improbable-AI/walk-these-ways)**
   — the original Multiplicity-of-Behavior / gait-conditioned RMA codebase
   (Margolis & Agrawal, CoRL 2023; paper ref [9]).
2. **Teddy-Liao/walk-these-ways-go2** — a community fork adapting the above
   to the Unitree Go2 robot.
3. **This author's fork/retraining** — further adapted and retrained for
   Go2, producing the checkpoint used here. The paper does not currently
   disclose this chain; it should, since the checkpoint is a self-trained
   artifact from a community fork, not an official upstream release.

## Algorithm and runner

| Parameter | Value |
|---|---|
| Algorithm | PPO (`clip_param=0.2`, `gamma=0.99`, `lam=0.95`, `entropy_coef=0.01`, `lr=1e-3`) |
| Runner | `RunnerArgs.algorithm_class_name = 'RMA'` (Rapid Motor Adaptation) |
| Recorded `max_iterations` (launch-time config) | 1500 |
| **Actual iterations reached** | **19,990** (≈3.26×10⁹ timesteps), per `outputs.log`. The saved config's `max_iterations=1500` does not describe what actually happened — the run was resumed/extended well past it. **Cite 19,990, not the config default, and note it was reached across possibly multiple resumed runs.** |

## Simulator and physics

| Parameter | Value |
|---|---|
| Simulator | Isaac Gym, PhysX backend, GPU pipeline (`use_gpu_pipeline=True`) |
| `sim.dt` | 0.005 s (200 Hz) |
| `physx.num_position_iterations` | 4 |
| `physx.num_velocity_iterations` | 0 |
| Gravity | `[0, 0, -9.81]` (matches the MuJoCo deployment scenes exactly) |

## Control

| Parameter | Value |
|---|---|
| Control decimation | 4 → 0.005 × 4 = 0.02 s = **50 Hz** policy rate (matches the 50 Hz used at deployment) |
| **Control type** | **`'actuator_net'`** — torque is computed by a learned actuator network, **not** a textbook PD law |
| Actuator net file | `resources/actuator_nets/unitree_go1.pt` — a **Go1** motor model, used unmodified for the Go2 |
| Nominal gains fed into that actuator net | `stiffness.joint=20.0`, `damping.joint=0.5` |

**This does not match deployment.** `harness.py` (and the other four
inference scripts) drive the Go2 with an idealized PD law using
`KP=25, KD=0.6` — different numbers from the training-time actuator-net
targets above, and a structurally different control law (a fixed PD
controller instead of a learned Go1 actuator-network-mediated torque
response). The paper's Section III.C claim that "all gains, scaling
factors, history length, and stepping frequency match the training
configuration" is not accurate for the gains and is not accurate about the
control law itself. See `AUDIT.md` bug list item 1 for the full analysis
and a proposed corrected sentence.

## Environment

| Parameter | Value |
|---|---|
| Parallel environments | 6800 |
| Episode length | 20 s (≈1001 steps) |
| Observation / action dimension | 70 / 12 |
| Terrain | `terrain_proportions=[0,0,0,0,0,0,0,0,1.0]` → **100% flat**, `curriculum=True` — corroborates the "flat-trained" claim used throughout Experiment 3 |

## Domain randomization

| Quantity | Range |
|---|---|
| Friction | `[0.1, 3.0]` |
| Restitution | `[0.0, 0.4]` |
| Added base mass | `[-1.0, 3.0]` kg |
| Motor strength | `[0.9, 1.1]` |
| Gravity | `±1.0`, resampled every 8 s |
| Lag timesteps | randomized, `lag_timesteps=6` |
| Push robots | **disabled** (`push_robots=False`) |
| Randomize Kp/Kd factor | **disabled** (`randomize_Kp_factor=False`, `randomize_Kd_factor=False`) — control gains were not randomized during training |

## Reward weights (nonzero terms only)

| Term | Weight |
|---|---|
| `tracking_lin_vel` | 1.0 |
| `tracking_ang_vel` | 0.5 |
| `tracking_contacts_shaped_force` | 4.0 |
| `tracking_contacts_shaped_vel` | 4.0 |
| `jump` | 10.0 |
| `dof_pos_limits` | -10.0 |
| `raibert_heuristic` | -10.0 |
| `orientation_control` | -5.0 |
| `collision` | -5.0 |
| `action_smoothness_1` | -0.1 |
| `action_smoothness_2` | -0.1 |
| `feet_slip` | -0.04 |
| `feet_clearance_cmd_linear` | -30.0 |
| plus small torque / dof-vel / dof-acc / lin-vel-z / ang-vel-xy penalties | (magnitude ≲0.01–0.1 each) |

The full `reward_scales` dict has ~40 keys; all keys not listed above are 0.

## Go2 model provenance (MJCF)

`stage2-go2-mujoco-inference/scenes/go2_model/go2.xml` is vendored from
Google DeepMind's `mujoco_menagerie`
(https://github.com/google-deepmind/mujoco_menagerie, `unitree_go2/go2.xml`).

Diffing the vendored copy against a fresh clone of `mujoco_menagerie`
(current HEAD `affef0836947b64cc06c4ab1cbf0152835693374`, 2026-04-16; the
file itself was last modified upstream at commit
`04674c8d0a4e12e33980a9e47fb604adc281fcc8`, 2025-06-17) shows **exactly one
line of difference**:

```
<   <compiler angle="radian" meshdir="assets" autolimits="true"/>   (upstream)
>   <compiler angle="radian" meshdir="." autolimits="true"/>         (vendored)
```

This is a path change only (the vendored copy flattens `assets/` into the
same directory as `go2.xml` for self-containment) — every physics
parameter, geom, joint, actuator, and all 16 mesh `.obj` files are
byte-identical to upstream (`cmp` verified on all 16 meshes). Since the
upstream file has not changed between 2025-06-17 and 2026-04-16, the
vendored copy can be pinned to any commit in that range; **the model
corresponds to `mujoco_menagerie` commit `04674c8d0a4e12e33980a9e47fb604adc281fcc8`
(2025-06-17) or later, through at least `affef083` (2026-04-16), with no
upstream changes in between.**

## Dependencies

See `requirements.txt` at the repo root. Pinned versions: `mujoco==3.8.0`,
`torch==2.11.0`, `numpy==2.2.6`, `gymnasium==1.2.3`, `glfw==2.10.0`,
Python 3.10.12.

## Checkpoint

`body_latest.jit` and `adaptation_module_latest.jit` are not committed to
this repository (binary artifacts, git-ignored). Fetch them with
`./download_policy.sh` from the repo's GitHub release (tag `v1.0-policy`);
the script verifies SHA-256 hashes against the values below.

| File | SHA-256 |
|---|---|
| `body_latest.jit` | `7b6e604e2147742a89ef50d91e7ee501023331b2589d1c3143a9d2ba858db7b5` |
| `adaptation_module_latest.jit` | `0e091f829dcfbedd4ccca6752863e1e2feca105f79da07d04e3545b8815dcc13` |
