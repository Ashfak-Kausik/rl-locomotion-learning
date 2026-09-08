"""
Experiment 5 -- Foot friction sensitivity sweep.

Follow-up to Experiment 4 (exp4_sensitivity_sweep.py), which swept the
FLOOR geom's friction/solref and found a hard-zero effect: the foot geoms
carry priority="1" (go2.xml:32) while the floor defaults to priority="0",
so MuJoCo's contact solver used the foot's declared friction unconditionally
regardless of what the floor's friction was set to (see SENSITIVITY_REPORT.md,
"Root cause: geom_priority" section). The foot's own friction is therefore
the live parameter for this question.

This script sweeps THAT parameter instead. go2.xml is never edited -- the
committed model keeps its declared foot friction of [0.8, 0.02, 0.01].
harness.run_trial() mutates model.geom_friction in memory for the four foot
geoms (FL/FR/RL/RR) after loading, before stepping, via the new
`foot_friction` kwarg (see harness.py). Verified directly beforehand
(see conversation / SENSITIVITY_REPORT.md) that the mutation now reaches
data.contact[i].friction unchanged from the override.

Sweep: foot sliding friction in {0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0},
torsional/rolling held at their declared values (0.02, 0.01). mu=0.8 is
the declared value and doubles as a control: it should reproduce exp1's
baseline (cmd_vx in {0.5, 1.0, 1.5}, seeds 0-14) exactly, since it is the
same physical configuration reached a different way (override vs. XML
default) -- this script checks that automatically and aborts loudly if it
does not match.

Commanded velocities {0.5, 1.0, 1.5} m/s, trot gait, n=15 seeds
(8 mu x 3 v x 15 seed = 360 trials).

Outputs (new files only):
  - results/exp5_foot_friction_sweep.csv
  - results/stats_exp5_foot_friction.csv   (written by exp5_report.py)
  - SENSITIVITY_REPORT.md extended with a new section (exp5_report.py)
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
FRICTION_VALUES = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]
DECLARED_MU = 0.8       # control condition -- must match go2.xml's own value
VELOCITIES = [0.5, 1.0, 1.5]

GAIT = "trot"
N_TRIALS = 15
SETTLE_S = 3.0
MEASURE_S = 30.0

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUT_CSV = os.path.join(RESULTS_DIR, "exp5_foot_friction_sweep.csv")

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
    print("EXPERIMENT 5 -- Foot Friction Sensitivity Sweep")
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
    print("Experiment 5 complete.")


if __name__ == "__main__":
    main()
