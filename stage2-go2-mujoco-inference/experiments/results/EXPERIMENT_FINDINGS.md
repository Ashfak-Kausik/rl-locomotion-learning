# Experiment Findings Log

This document records the interpreted results of every experiment, written
immediately after each run while the analysis is fresh. It is the primary
reference for drafting the paper. Raw per-trial data is in the corresponding
`results/*.csv` files. This log is the "what it means" layer; the CSVs are
the "what happened" layer.

**Setup (constant across all experiments unless noted):**
- Policy: walk-these-ways pretrained (Isaac Gym), `body_latest.jit` + `adaptation_module_latest.jit`
- Simulator: MuJoCo, CPU only (no GPU)
- Robot: Unitree Go2 (mujoco_menagerie MJCF, self-contained copy in `scenes/go2_model/`)
- Control: policy 50 Hz, sim 500 Hz, decimation 10, PD gains kp=25 kd=0.6
- Default pose: training defaults (FL/RL hips +0.1, FR/RR hips −0.1, etc.)
- Initial-condition randomization (per seed): joint pos ±0.02 rad, base
  height ±0.01 m, base yaw ±0.05 rad, joint vel ±0.05 rad/s
- Trials per condition: 5 (seeds 0–4), 3 s settle + 30 s measurement
- Fall criteria: body height < 0.15 m OR tilt > 60° from upright

---

## Experiment 1 — Velocity Sweep (flat ground, trot)

**Run date:** 2026-05-18
**Script:** `exp1_velocity_sweep.py`
**Raw data:** `results/exp1_velocity_sweep.csv`
**Conditions:** commanded vx ∈ {0.0, 0.25, 0.5, 0.75, 1.0, 1.5} m/s

### Results table (mean ± std over 5 trials)

| cmd_vx | survival | achieved vx | track err | lateral drift | height std |
|--------|----------|-------------|-----------|---------------|------------|
| 0.00 | 100% | 0.073±0.001 | 0.073±0.001 | 0.043±0.001 | 0.007±0.000 |
| 0.25 | 100% | 0.145±0.003 | 0.105±0.003 | 0.111±0.004 | 0.006±0.000 |
| 0.50 | 100% | 0.227±0.005 | 0.273±0.005 | 0.162±0.006 | 0.006±0.000 |
| 0.75 | 100% | 0.367±0.004 | 0.383±0.004 | 0.134±0.011 | 0.006±0.000 |
| 1.00 | 100% | 0.504±0.001 | 0.496±0.001 | 0.018±0.010 | 0.007±0.000 |
| 1.50 | 100% | 0.558±0.010 | 0.942±0.010 | 0.347±0.015 | 0.006±0.000 |

### Findings

**F1.1 — Systematic velocity under-tracking (core sim-to-sim gap).**
The policy achieves only ~37–55% of commanded forward velocity. Tracking
error grows monotonically with command: 0.073 → 0.942 m/s. At cmd=1.5 m/s
the robot delivers only 0.558 m/s (error 0.942). Headline result: Isaac-Gym
→ MuJoCo transfer does NOT preserve velocity-tracking fidelity, and the gap
widens with commanded speed.

**F1.2 — Velocity saturation (~0.55 m/s ceiling).**
Achieved velocity plateaus. cmd 1.0 → 1.5 yields achieved 0.504 → 0.558
only. The policy has an effective max speed in MuJoCo far below its
presumed Isaac Gym capability. Suggests the transfer bottleneck is dynamic,
not just a constant offset.

**F1.3 — Directional-stability sweet spot at cmd ≈ 1.0 m/s.**
Lateral drift is non-monotonic: 0.043, 0.111, 0.162, 0.134, **0.018**,
0.347. Pronounced minimum at cmd=1.0 (near-zero drift) with higher drift
both below and above. HYPOTHESIS: the policy's gait is most symmetric at
the operating point its internal dynamics were tuned around; away from it,
sim-to-sim contact asymmetries express as lateral drift. Most novel/
non-obvious finding — flag for discussion section.

**F1.4 — Posture transfers even when velocity does not.**
Body-height std ≈ 0.006–0.007 across ALL speeds (dead flat). The policy
cannot track velocity in MuJoCo but holds posture rock-solid. Clean
decoupling: proprioceptive stabilization transfers well; velocity tracking
transfers poorly. Citable distinction.

**F1.5 — Variance grows with commanded velocity.**
Std rises 0.001 (cmd=0) → 0.010 (cmd=1.5). The policy becomes more
sensitive to initial conditions at higher speeds. Higher speed = less
predictable transfer.

**F1.6 — 100% flat-ground survival at all speeds.**
No falls on flat ground anywhere. Establishes the control baseline so that
Experiment 3 terrain failures are attributable to terrain, not policy
instability.

### Paper usage
- This is **Table 1** (sim-to-sim velocity transfer).
- A figure: achieved-vs-commanded velocity with the y=x ideal line overlaid,
  plus a second axis or panel for drift showing the cmd=1.0 minimum.
- F1.1, F1.2 → Results subsection "Velocity tracking gap"
- F1.3 → Discussion (hypothesis-driven, the interesting one)
- F1.4 → Results subsection "Posture vs. tracking decoupling"

---

## Experiment 2 — Gait Robustness (flat ground, cmd_vx = 0.5 m/s)

**Run date:** 2026-05-18
**Script:** `exp2_gait_robustness.py`
**Raw data:** `results/exp2_gait_robustness.csv`
**Conditions:** gait ∈ {trot, pace, bound}, fixed cmd_vx = 0.5 m/s

### Results table (mean ± std over 5 trials)

| gait | survival | achieved vx | track err | lateral drift | height std |
|------|----------|-------------|-----------|---------------|------------|
| trot | 100% | 0.227±0.005 | 0.273±0.005 | 0.162±0.006 | 0.006±0.000 |
| pace | 100% | 0.327±0.002 | 0.173±0.002 | 0.087±0.006 | 0.011±0.000 |
| bound | 100% | 0.167±0.006 | 0.333±0.006 | 0.203±0.002 | 0.002±0.000 |

### Findings

**F2.1 — Gait-conditioning transfers robustly for ALL gaits (survival).**
Contrary to the expectation (from informal interactive testing) that pace
and bound would be unstable, all three gaits achieved 100% survival over
5 trials at cmd=0.5 m/s on flat ground. The gait-conditioning mechanism
itself survives Isaac-Gym → MuJoCo transfer.

**F2.2 — Pace transfers BEST at moderate speed, not trot.**
Counterintuitive headline finding. At cmd=0.5 m/s, pace achieves the
highest forward velocity (0.327 vs trot 0.227), the lowest tracking error
(0.173 vs trot 0.273), and the lowest lateral drift (0.087 vs trot 0.162).
The policy's nominal "default" gait (trot) is NOT the best-transferring
gait under MuJoCo physics at this speed. Overturns the naive assumption.

**F2.3 — Bound trades forward progress for posture rigidity.**
Bound has the lowest achieved velocity (0.167) and highest tracking error
(0.333), but by far the most stable body height (h_std 0.002, ~3× more
stable than trot's 0.006). Coherent physical story: bound is a stiff,
hopping-style gait — poor forward efficiency, very rigid vertical posture.

**F2.4 — Gaits occupy distinct points on a performance tradeoff surface.**
Pace → best tracking + least drift. Trot → middling at everything.
Bound → worst tracking, best posture rigidity. Gait choice is a
multi-objective tradeoff under sim-to-sim transfer, not a strict ordering.

### Caveat / limitation
Tested only at cmd=0.5 m/s. Pace's advantage may not hold across the
velocity range (cf. Exp 1, where trot's drift was minimal at cmd=1.0).
A full gait × velocity grid is future work.

### Paper usage
- This is **Table 2** (gait robustness under transfer).
- F2.2 is the surprising/headline result → Discussion section.
- Pair with Exp 1: together they show transfer quality depends on BOTH
  commanded velocity AND gait, in non-obvious ways.
- Honest limitation (single speed) strengthens credibility — state it.


---

## Experiment 3 — Terrain Robustness
*(to be filled after run)*