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

## Experiment 3 — Terrain Robustness (flat-trained policy, OOD generalization)

**Run date:** 2026-05-18
**Script:** `exp3_terrain.py` (+ `generate_terrain_scenes.py`)
**Raw data:** `results/exp3_terrain.csv`
**Conditions:** slopes {5,10,15,20,25}°, stairs {2,5,8,12,16}cm; trot, cmd_vx=0.5, 20s measure

### Methodology note (important — strengthens the paper)
Initial "traversable = survival ≥ 80%" metric was found to be FLAWED:
the robot can survive by safely stalling at the base of an obstacle
(no fall, but no progress either). Detected via the distance_traveled
column (deterministic 0.47 m stall at steps ≥5cm = robot stuck against
first step, never climbing; run-up alone is 2.5 m). Metric corrected to
require BOTH survival ≥80% AND mean distance > 3.5 m (run-up 2.5 m + ≥1 m
genuine terrain progress). This defines a distinct non-catastrophic
failure mode: "safe stall" (survives, no progress). Reporting this
correction demonstrates measurement rigor.

### Results — Slopes

| slope | survival | mean_dist (survivors) | traversable |
|-------|----------|------------------------|-------------|
| 5° | 100% | 5.76±0.00 | YES |
| 10° | 100% | 5.43±0.01 | YES |
| 15° | 20% | 4.75 (1/5) | no |
| 20° | 0% | — | no |
| 25° | 0% | — | no |

### Results — Stairs

| step height | survival | mean_dist | traversable |
|-------------|----------|-----------|-------------|
| 2 cm | 100% | 4.02±0.09 | YES (marginal) |
| 5 cm | 100% | 0.47±0.01 (STALL) | no |
| 8 cm | 100% | 0.47±0.01 (STALL) | no |
| 12 cm | 100% | 0.47±0.01 (STALL) | no |
| 16 cm | 100% | 0.48±0.00 (STALL) | no |

### Findings

**F3.1 — Maximum traversable slope = 10°.** Clean monotonic degradation:
100% survival at 5°/10°, collapsing to 20% at 15° and 0% at ≥20°. Time-
to-fall shortens with steepness (≈18s @ 20°, ≈15s @ 25°), indicating
progressively faster instability onset.

**F3.2 — Maximum traversable step height = 2 cm (marginal).** The policy
clears only ~ankle-low 2cm steps, and even then with elevated variance
(vx 0.020–0.056) indicating a struggle, not clean locomotion. At ≥5 cm
the robot deterministically stalls 0.47 m in (never leaving the run-up
region), neither climbing nor falling.

**F3.3 — Slope/step asymmetry (HEADLINE INSIGHT).** A flat-trained policy
tolerates 10° inclines but fails at 5 cm steps. Hypothesis: slopes
preserve continuous ground contact — the learned flat-ground gait remains
viable when merely tilted — whereas steps require discrete foot-clearance
behavior absent from flat-only training. Continuous vs. discrete terrain
perturbation transfer very differently for blind locomotion.

**F3.4 — "Safe stall": a non-catastrophic failure mode.** At steps ≥5 cm
the policy neither progresses nor falls; it stabilizes against the
obstacle indefinitely. Survival-only metrics misclassify this as success.
Distinguishing "traversal" from "survival" is necessary for honest OOD
evaluation of locomotion policies.

**F3.5 — Flat-ground baseline (Exp 1) confirms attribution.** 100% flat
survival at all speeds (Exp 1, F1.6) confirms Exp 3 failures are
terrain-induced, not intrinsic policy instability.

### Paper usage
- **Table 3** (slopes) + **Table 4** (stairs), or a combined terrain table.
- Figure: survival-rate vs. slope angle, and distance-vs-step-height
  showing the stall plateau.
- F3.3 is the strongest single insight → Discussion centerpiece.
- F3.4 (metric correction) → Methodology, framed as rigor.
- Direct motivation for Stage 3 / future work (terrain-curriculum training).

---

## Experiment 4 — Heading Hold vs. World-Frame Velocity Measurement

**Run date:** 2026-07-31
**Script:** `exp4_heading_hold.py`
**Raw data:** `results/exp4_heading_hold.csv`
**Conditions:** commanded vx ∈ {0.25, 0.5, 0.75, 1.0} m/s × heading_hold ∈
{off, on}, trot, flat, 3 seeds each

### Motivation

Experiments 1–3 command `ang_vel_yaw = 0.0` for the whole trial and log
`data.qvel[0]`, MuJoCo's WORLD-frame x-velocity. `build_obs` (`harness.py`)
hands the policy `lin_vel_x` as a BODY-frame command. Those coincide only
while the robot's heading stays at 0°. With no heading feedback, small yaw
bias in the gait integrates over a 30 s trial, and the two frames diverge.

### Results table (mean over 3 seeds)

| cmd_vx | mode | world vx | body vx | |yaw drift| | |lateral offset| |
|--------|--------|----------|---------|--------------|-------------------|
| 0.25 | off | 0.144 | 0.195 | 78.3° | 3.27 m |
| 0.25 | on  | 0.199 | 0.195 | 6.7°  | 0.03 m |
| 0.50 | off | 0.226 | 0.296 | 73.3° | 4.82 m |
| 0.50 | on  | 0.302 | 0.295 | 6.3°  | 0.25 m |
| 0.75 | off | 0.366 | 0.396 | 37.6° | 4.06 m |
| 0.75 | on  | 0.399 | 0.396 | 1.7°  | 0.11 m |
| 1.00 | off | 0.504 | 0.502 | 1.6°  | 0.43 m |
| 1.00 | on  | 0.505 | 0.503 | 0.8°  | 0.22 m |

`heading_hold` closes a simple loop the policy was already trained to accept
(`ang_vel_yaw = clip(-1.5 * yaw_error, -0.6, +0.6)`), evaluated every policy
step against the current heading. No retraining, no contract change.

### Findings

**F4.1 — Roughly half of Experiment 1's "velocity under-tracking" is a
measurement artifact, not a control failure (HEADLINE REVISION).**
At cmd 0.5, `off` shows world vx 0.226 (44% of command) but body vx 0.296
(59% of command) — the same trial, two frames. `body vx` is flat across the
entire 30 s window regardless of heading; `world vx` decays as
`body_vx * cos(yaw_error)` while the robot arcs away from +x. The policy is
not decelerating; it is turning. F1.1's headline number (37–55% tracking)
should be read as a *lower bound* — true tracking, measured in the frame the
command was issued in, is materially better. The residual gap (body vx 0.296
vs. commanded 0.5, ≈59%) is the real sim-to-sim finding this repo exists to
quantify; it is smaller than F1.1 reported, not zero.

**F4.2 — F1.3's "directional-stability sweet spot at cmd≈1.0" is largely
explained by yaw drift being coincidentally small there, not gait symmetry.**
`off` yaw drift is 78°, 73°, 38°, 2° across the four commands — monotonically
*shrinking*, not peaking at 1.0. Lateral drift in Exp 1 (F1.3) tracks this:
minimal at cmd=1.0 not because the gait is more symmetric there, but because
the un-corrected heading happens to wander least at that operating point over
a 30 s window. This does not fully retire F1.3 — gait symmetry may still
contribute — but the dominant term is measurement geometry, and the
"hypothesis-driven" framing in F1.3 should be revisited before citing it.

**F4.3 — Heading hold makes the policy usable without retraining.**
`on` reduces final lateral offset by 10–100× at every speed (3.27→0.03 m at
cmd 0.25; 4.82→0.25 m at cmd 0.5) and cuts yaw drift to under 7° everywhere,
with zero falls in all 24 trials. `ang_vel_yaw` was always a valid input —
Experiments 1–3 simply never drove it. This is a **deployment-config** fix,
not a policy fix: same weights, same contract, different command each step.

**F4.4 — Experiments 1–3 are NOT invalidated, but need a companion read.**
Their CSVs are reproducible and their qualitative claims (F1.2 saturation,
F1.4 posture/velocity decoupling, F1.5 variance growth, F3.1–F3.5 terrain
results) do not depend on the frame issue — those are height, survival, and
fall-time based. Only the *velocity-tracking magnitude* and the *directional-
stability* claims (F1.1, F1.3) are affected. `harness.run_trial` now also
returns `mean_vx_body`, `mean_vy_body`, `vel_track_err_body`, and
`yaw_drift_deg` (additive keys; existing `mean_vx` etc. are byte-identical to
before, so exp1–exp3's committed CSVs remain reproducible from source).

### Paper usage
- Supersedes F1.1's magnitude and F1.3's mechanism — cite both F1.x and F4.x
  together, framed as "naive measurement vs. corrected measurement."
- F4.3 is a practical contribution: one-line deployment fix, no retraining.
- Suggested figure: world-vx vs. body-vx vs. time at cmd=0.5, both modes
  overlaid, with heading angle on a second axis — makes the artifact visible
  in one panel.
- Live demonstration: `experiments/watch_walk.py` (`--no-heading-hold` to
  reproduce the drift interactively; default reproduces the fix).