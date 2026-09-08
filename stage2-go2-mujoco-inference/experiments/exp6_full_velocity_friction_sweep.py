"""
Experiment 6 -- Full velocity sweep x foot friction (does saturation survive?).

Direct follow-up to Experiment 5 (exp5_foot_friction_sweep.py), which swept
foot friction at three commanded velocities {0.5, 1.0, 1.5} and found that at
cmd=1.5, mu=1.5 reaches 0.6642 m/s -- well above the paper's stated ~0.55 m/s
saturation ceiling, which was measured at the declared mu=0.8. That result
only covered the top of the velocity range. This experiment asks the
question directly at every commanded velocity the paper actually reports:
does the achieved-vs-commanded curve flatten at every friction value, or only
at mu=0.8?

Sweep: the FULL exp1 velocity grid {0.0, 0.25, 0.5, 0.75, 1.0, 1.5} m/s,
crossed with foot friction mu in {0.4, 0.8 (declared, control), 1.5}. Trot
gait, n=15 seeds/condition (3 mu x 6 v x 15 seed = 270 trials).

Mechanism (unchanged from exp5): go2.xml is never edited. harness.run_trial()
mutates model.geom_friction in memory for the four foot geoms (FL/FR/RL/RR)
via the `foot_friction` kwarg. The floor's friction is structurally inert
(foot priority="1", see SENSITIVITY_REPORT.md Part 1) so only the foot value
is varied.

Control check (required before trusting anything else): the mu=0.8 condition
must reproduce exp1_velocity_sweep_30seed.csv's seeds 0-14 at all six
commanded velocities, since it is the same physical configuration reached via
override instead of the XML default. Aborts loudly if it does not match.

Outputs (new files only):
  - results/exp6_full_velocity_friction_sweep.csv
  - results/stats_exp6_full_velocity_friction.csv  (written by exp6_report.py)
  - SENSITIVITY_REPORT.md extended with a new section (exp6_report.py)
"""

import os
import csv
import sys
import time
import numpy as np

from harness import run_trials_parallel, physical_core_count

SCENE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "scenes", "go2_flat.xml")
)

FOOT_TORSIONAL = 0.02   # declared value, go2.xml:33 -- held fixed
FOOT_ROLLING = 0.01     # declared value, go2.xml:33 -- held fixed
FRICTION_VALUES = [0.4, 0.8, 1.5]
DECLARED_MU = 0.8       # control condition -- must match go2.xml's own value
VELOCITIES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]

GAIT = "trot"
N_TRIALS = 15
SETTLE_S = 3.0
MEASURE_S = 30.0

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUT_CSV = os.path.join(RESULTS_DIR, "exp6_full_velocity_friction_sweep.csv")

# Baseline to check the mu=0.8 control against: exp1's own 30-seed CSV,
# restricted to seeds 0-14 (this sweep's n=15) for an apples-to-apples
# comparison. Read-only.
EXP1_CSV = os.path.join(RESULTS_DIR, "exp1_velocity_sweep_30seed.csv")

CSV_FIELDS = [
    "foot_friction_mu", "foot_friction_tor", "foot_friction_roll",
    "scene", "cmd_vx", "cmd_vy", "cmd_yaw", "gait", "seed",
    "fell", "fall_time_s", "mean_vx", "mean_vy", "vel_track_err",
    "lateral_drift", "height_mean", "height_std", "distance_traveled",
]


def load_exp1_baseline():
    with open(EXP1_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    baseline = {}
    for v in VELOCITIES:
        sub = [r for r in rows if r["cmd_vx"] == str(v) and r["fell"] == "False"
               and int(r["seed"]) < N_TRIALS]
        vx = np.array([float(r["mean_vx"]) for r in sub])
        baseline[v] = float(vx.mean())
    return baseline


def build_jobs():
    jobs, meta = [], []
    for mu in FRICTION_VALUES:
        for v in VELOCITIES:
            for seed in range(N_TRIALS):
                jobs.append(dict(
                    scene_path=SCENE, lin_vel_x=v, lin_vel_y=0.0, ang_vel_yaw=0.0,
                    gait=GAIT, settle_s=SETTLE_S, measure_s=MEASURE_S, seed=seed,
                    foot_friction=(mu, FOOT_TORSIONAL, FOOT_ROLLING),
                ))
                meta.append(mu)
    return jobs, meta


def main():
    jobs, meta = build_jobs()
    n_workers = physical_core_count()

    print("=" * 70)
    print("EXPERIMENT 6 -- Full Velocity Sweep x Foot Friction")
    print("=" * 70)
    print("Mutated geoms: FL, FR, RL, RR only (go2_flat.xml + go2.xml model, in-memory)")
    print("go2.xml is NOT edited -- committed foot friction stays [0.8, 0.02, 0.01].")
    print(f"Sweep: mu in {FRICTION_VALUES} x v in {VELOCITIES} x {N_TRIALS} seeds "
          f"= {len(jobs)} trials")
    print(f"Workers: {n_workers} (physical cores)")
    print("=" * 70)

    t0 = time.time()
    print("Running trials in parallel...")
    rows = run_trials_parallel(jobs, n_workers=n_workers)
    t1 = time.time()
    print(f"Done in {t1 - t0:.1f}s ({(t1 - t0) / len(jobs):.2f}s/trial effective).\n")

    for row, mu in zip(rows, meta):
        row["foot_friction_mu"] = mu
        row["foot_friction_tor"] = FOOT_TORSIONAL
        row["foot_friction_roll"] = FOOT_ROLLING

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT_CSV}\n")

    # ------------------------------------------------------------------
    # Control check: mu=0.8 condition must reproduce exp1's baseline
    # (same physical config, reached via override instead of XML default).
    # ------------------------------------------------------------------
    print("=" * 70)
    print("CONTROL CHECK: mu=0.8 (declared value) vs exp1 baseline (seeds 0-14)")
    print("=" * 70)
    exp1_baseline = load_exp1_baseline()
    control_ok = True
    for v in VELOCITIES:
        cond = [r for r in rows if r["foot_friction_mu"] == DECLARED_MU
                and r["cmd_vx"] == v and not r["fell"]]
        vx = np.array([r["mean_vx"] for r in cond])
        got = float(vx.mean())
        want = exp1_baseline[v]
        diff = abs(got - want)
        status = "MATCH" if diff < 1e-6 else "MISMATCH"
        if diff >= 1e-6:
            control_ok = False
        print(f"  cmd_vx={v}: override-run mean_vx={got:.6f}  exp1 baseline={want:.6f}  "
              f"diff={diff:.2e}  [{status}]")

    if not control_ok:
        print("\n*** CONTROL CHECK FAILED. The mu=0.8 override condition does not "
              "reproduce exp1's baseline. Per instructions, stopping here -- do not "
              "trust the sweep results below until this is resolved. ***")
        sys.exit(1)
    print("\nControl check passed: mu=0.8 override reproduces exp1 exactly.")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Quick console summary
    # ------------------------------------------------------------------
    print("\nSUMMARY (mean achieved vx +/- std, survival)")
    header = f"{'mu':>5} | " + " | ".join(f"v={v:.2f}".rjust(18) for v in VELOCITIES)
    print(header)
    print("-" * len(header))
    for mu in FRICTION_VALUES:
        cells = []
        for v in VELOCITIES:
            cond_rows = [r for r in rows if r["foot_friction_mu"] == mu and r["cmd_vx"] == v]
            survived = [r for r in cond_rows if not r["fell"]]
            if not survived:
                cells.append("ALL FELL".rjust(18))
                continue
            vx = np.array([r["mean_vx"] for r in survived])
            surv_rate = len(survived) / len(cond_rows)
            cells.append(f"{vx.mean():.4f}+/-{vx.std():.4f} {surv_rate:>4.0%}".rjust(18))
        print(f"{mu:>5.1f} | " + " | ".join(cells))

    print("=" * 70)
    print("Experiment 6 complete.")


if __name__ == "__main__":
    main()
