"""
Experiment 4 -- Contact-parameter sensitivity sweep (friction / stiffness).

Research question (from two reviewer comments on the sim-to-sim transfer
paper): how much of the measured velocity gap between commanded and
achieved forward speed is attributable to *uncalibrated MuJoCo contact
parameters* rather than genuine Isaac-Gym-vs-MuJoCo simulator difference?

AUDIT.md Part A.5 established that the floor/ramp/step geoms in every
scene declare no <geom friction> or <geom solref> at all, so they run at
MuJoCo's built-in defaults (mu=1.0, solref=[0.02, 1]) -- values nobody
ever reasoned about -- while the Isaac Gym training config randomized
ground friction over [0.1, 3.0]. This script probes that gap directly.

Method: go2_flat.xml is loaded exactly as committed; NO scene file is
created or edited. Before each trial, harness.run_trial() mutates
model.geom_friction / model.geom_solref *in memory* for the geom named
"floor" only (see floor_friction / floor_solref kwargs added to
run_trial in harness.py). Foot friction, set explicitly in go2.xml
(class="foot", friction="0.8 0.02 0.01"), is never touched -- the
mutation targets a different geom by name.

Task 1 -- friction sweep: ground sliding friction in
  {0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0}
(spans the robot's own foot value 0.8, the current unexamined default
1.0, and the extremes of the Isaac Gym randomization range 0.1-3.0;
0.1 itself is omitted as it is degenerate/near-frictionless and not
informative for a walking gait). Torsional/rolling friction held at the
current default (0.005, 0.0001) throughout -- only sliding friction
varies. Commanded velocities {0.5, 1.0, 1.5} m/s, trot gait, n=15 seeds.

Task 2 -- contact stiffness sweep: floor solref time-constant in
  {0.005, 0.01, 0.02, 0.04}
(dampratio held at the default 1.0), sliding friction held at 1.0
(current default). Commanded velocity 1.0 m/s only, trot gait, n=15.

Outputs (new files, nothing existing is touched):
  - results/exp4_friction_sweep.csv     (315 rows: 7 friction x 3 vel x 15 seed)
  - results/exp4_stiffness_sweep.csv    (60 rows: 4 solref x 1 vel x 15 seed)
  - results/stats_exp4_friction.csv     (mean/SD/95% CI per friction x vel)
  - results/stats_exp4_stiffness.csv    (mean/SD/95% CI per solref)
  - SENSITIVITY_REPORT.md               (written by exp4_report.py, not here)
"""

import os
import csv
import time
import numpy as np

from harness import run_trials_parallel, physical_core_count

SCENE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "scenes", "go2_flat.xml")
)

# Torsional / rolling friction held fixed at MuJoCo's current defaults
# throughout Task 1 -- only sliding friction (first component) varies.
FRICTION_TORSIONAL = 0.005
FRICTION_ROLLING = 0.0001
FRICTION_VALUES = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
FRICTION_VELOCITIES = [0.5, 1.0, 1.5]

# solref dampratio held fixed at the current default (1.0) throughout
# Task 2 -- only the time constant varies.
SOLREF_DAMPRATIO = 1.0
SOLREF_TIMECONSTS = [0.005, 0.01, 0.02, 0.04]
SOLREF_FIXED_FRICTION_MU = 1.0  # current default, held fixed for Task 2
SOLREF_VELOCITY = 1.0

GAIT = "trot"
N_TRIALS = 15
SETTLE_S = 3.0
MEASURE_S = 30.0

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FRICTION_CSV = os.path.join(RESULTS_DIR, "exp4_friction_sweep.csv")
STIFFNESS_CSV = os.path.join(RESULTS_DIR, "exp4_stiffness_sweep.csv")

BASE_FIELDS = [
    "scene", "cmd_vx", "cmd_vy", "cmd_yaw", "gait", "seed",
    "fell", "fall_time_s", "mean_vx", "mean_vy", "vel_track_err",
    "lateral_drift", "height_mean", "height_std", "distance_traveled",
]
FRICTION_FIELDS = ["friction_mu", "friction_tor", "friction_roll"] + BASE_FIELDS
STIFFNESS_FIELDS = ["solref_tconst", "solref_dampratio"] + BASE_FIELDS


def build_friction_jobs():
    jobs = []
    meta = []  # parallel list of (mu,) for annotating results after the fact
    for mu in FRICTION_VALUES:
        for v in FRICTION_VELOCITIES:
            for seed in range(N_TRIALS):
                jobs.append(dict(
                    scene_path=SCENE, lin_vel_x=v, lin_vel_y=0.0, ang_vel_yaw=0.0,
                    gait=GAIT, settle_s=SETTLE_S, measure_s=MEASURE_S, seed=seed,
                    floor_friction=(mu, FRICTION_TORSIONAL, FRICTION_ROLLING),
                ))
                meta.append(mu)
    return jobs, meta


def build_stiffness_jobs():
    jobs = []
    meta = []
    for tc in SOLREF_TIMECONSTS:
        for seed in range(N_TRIALS):
            jobs.append(dict(
                scene_path=SCENE, lin_vel_x=SOLREF_VELOCITY, lin_vel_y=0.0,
                ang_vel_yaw=0.0, gait=GAIT, settle_s=SETTLE_S, measure_s=MEASURE_S,
                seed=seed,
                floor_friction=(SOLREF_FIXED_FRICTION_MU, FRICTION_TORSIONAL, FRICTION_ROLLING),
                floor_solref=(tc, SOLREF_DAMPRATIO),
            ))
            meta.append(tc)
    return jobs, meta


def main():
    friction_jobs, friction_meta = build_friction_jobs()
    stiffness_jobs, stiffness_meta = build_stiffness_jobs()
    total_trials = len(friction_jobs) + len(stiffness_jobs)

    n_workers = physical_core_count()
    print("=" * 70)
    print("EXPERIMENT 4 -- Contact-Parameter Sensitivity Sweep")
    print("=" * 70)
    print("Mutated geom: 'floor' only (go2_flat.xml, in-memory, no new scene files)")
    print("Foot geom friction (class=\"foot\", go2.xml) is NOT touched by this script.")
    print(f"Task 1 (friction): mu in {FRICTION_VALUES} x v in {FRICTION_VELOCITIES} "
          f"x {N_TRIALS} seeds = {len(friction_jobs)} trials")
    print(f"Task 2 (stiffness): solref_tconst in {SOLREF_TIMECONSTS} "
          f"x v={SOLREF_VELOCITY} x {N_TRIALS} seeds = {len(stiffness_jobs)} trials")
    print(f"Total trials: {total_trials}")
    print(f"Workers: {n_workers} (physical cores)")
    print("=" * 70)

    t0 = time.time()
    print("Running Task 1 (friction sweep) in parallel...")
    friction_rows = run_trials_parallel(friction_jobs, n_workers=n_workers)
    t1 = time.time()
    print(f"Task 1 done in {t1 - t0:.1f}s ({(t1 - t0) / len(friction_jobs):.2f}s/trial "
          f"effective).\n")

    print("Running Task 2 (stiffness sweep) in parallel...")
    stiffness_rows = run_trials_parallel(stiffness_jobs, n_workers=n_workers)
    t2 = time.time()
    print(f"Task 2 done in {t2 - t1:.1f}s ({(t2 - t1) / len(stiffness_jobs):.2f}s/trial "
          f"effective).\n")
    print(f"Total wall time: {t2 - t0:.1f}s\n")

    # Annotate rows with the sweep parameter (run_trial's return dict does not
    # carry it -- it's a job-level condition, not something derivable from the
    # trial outcome).
    for row, mu in zip(friction_rows, friction_meta):
        row["friction_mu"] = mu
        row["friction_tor"] = FRICTION_TORSIONAL
        row["friction_roll"] = FRICTION_ROLLING

    for row, tc in zip(stiffness_rows, stiffness_meta):
        row["solref_tconst"] = tc
        row["solref_dampratio"] = SOLREF_DAMPRATIO

    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(FRICTION_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FRICTION_FIELDS)
        w.writeheader()
        w.writerows(friction_rows)
    print(f"Wrote {len(friction_rows)} rows to {FRICTION_CSV}")

    with open(STIFFNESS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=STIFFNESS_FIELDS)
        w.writeheader()
        w.writerows(stiffness_rows)
    print(f"Wrote {len(stiffness_rows)} rows to {STIFFNESS_CSV}\n")

    # ------------------------------------------------------------------
    # Quick console summary
    # ------------------------------------------------------------------
    print("=" * 70)
    print("TASK 1 SUMMARY (mean achieved vx +/- std, survival)")
    print("=" * 70)
    header = f"{'mu':>5} | " + " | ".join(f"v={v:.2f}".rjust(16) for v in FRICTION_VELOCITIES)
    print(header)
    print("-" * len(header))
    for mu in FRICTION_VALUES:
        cells = []
        for v in FRICTION_VELOCITIES:
            rows = [r for r in friction_rows if r["friction_mu"] == mu and r["cmd_vx"] == v]
            survived = [r for r in rows if not r["fell"]]
            if not survived:
                cells.append("ALL FELL".rjust(16))
                continue
            vx = np.array([r["mean_vx"] for r in survived])
            surv_rate = len(survived) / len(rows)
            cells.append(f"{vx.mean():.3f}+/-{vx.std():.3f} {surv_rate:>4.0%}".rjust(16))
        print(f"{mu:>5.1f} | " + " | ".join(cells))

    print()
    print("=" * 70)
    print(f"TASK 2 SUMMARY (mean achieved vx +/- std, survival) at cmd_vx={SOLREF_VELOCITY}")
    print("=" * 70)
    for tc in SOLREF_TIMECONSTS:
        rows = [r for r in stiffness_rows if r["solref_tconst"] == tc]
        survived = [r for r in rows if not r["fell"]]
        surv_rate = len(survived) / len(rows)
        if not survived:
            print(f"tconst={tc:.3f}: ALL FELL")
            continue
        vx = np.array([r["mean_vx"] for r in survived])
        print(f"tconst={tc:.3f}: vx={vx.mean():.3f}+/-{vx.std():.3f} survival={surv_rate:.0%}")

    print("=" * 70)
    print("Experiment 4 complete.")


if __name__ == "__main__":
    main()
