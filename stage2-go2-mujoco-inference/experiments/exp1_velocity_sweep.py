"""
Experiment 1 — Sim-to-Sim Performance Gap (flat ground).

Research question: How faithfully does the Isaac-Gym-trained walk-these-ways
policy reproduce commanded forward velocity when deployed in MuJoCo on CPU?

Protocol:
  - Terrain: flat
  - Gait: trot
  - Commanded forward velocities: {0.0, 0.25, 0.5, 0.75, 1.0, 1.5} m/s
  - 5 trials per velocity (seeds 0-4)
  - 3 s settle + 30 s measurement window

Outputs:
  - results/exp1_velocity_sweep.csv   (one row per trial)
  - Printed summary table (mean ± std per velocity)
"""

import os
import csv
import numpy as np
import torch

from harness import run_trial, POLICY_DIR

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
SCENE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "scenes", "go2_flat.xml")
)
VELOCITIES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
N_TRIALS = 5
SETTLE_S = 3.0
MEASURE_S = 30.0
RESULTS_CSV = os.path.join(os.path.dirname(__file__), "results",
                           "exp1_velocity_sweep.csv")

CSV_FIELDS = [
    "scene", "cmd_vx", "cmd_vy", "cmd_yaw", "gait", "seed",
    "fell", "fall_time_s", "mean_vx", "mean_vy", "vel_track_err",
    "lateral_drift", "height_mean", "height_std", "distance_traveled",
]


def main():
    print("=" * 70)
    print("EXPERIMENT 1 — Velocity Sweep (flat ground, trot)")
    print("=" * 70)
    print(f"Velocities: {VELOCITIES} m/s")
    print(f"Trials per velocity: {N_TRIALS}")
    print(f"Settle: {SETTLE_S}s | Measure: {MEASURE_S}s")
    print(f"Total trials: {len(VELOCITIES) * N_TRIALS}")
    print("=" * 70)

    # Load policy networks ONCE and reuse across all trials (much faster)
    print("Loading policy networks...")
    body_net = torch.jit.load(f"{POLICY_DIR}/body_latest.jit")
    body_net.eval()
    adapt_net = torch.jit.load(f"{POLICY_DIR}/adaptation_module_latest.jit")
    adapt_net.eval()
    print("Policy loaded.\n")

    all_rows = []

    for v in VELOCITIES:
        print(f"--- Commanded velocity: {v:.2f} m/s ---")
        for seed in range(N_TRIALS):
            result = run_trial(
                SCENE,
                lin_vel_x=v, lin_vel_y=0.0, ang_vel_yaw=0.0,
                gait="trot",
                settle_s=SETTLE_S, measure_s=MEASURE_S,
                body_net=body_net, adapt_net=adapt_net,
                seed=seed,
            )
            all_rows.append(result)
            status = "FELL @ %.1fs" % result["fall_time_s"] if result["fell"] \
                else f"vx={result['mean_vx']:.3f} drift={result['lateral_drift']:.3f}"
            print(f"  seed {seed}: {status}")
        print()

    # Write CSV
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"Wrote {len(all_rows)} rows to {RESULTS_CSV}\n")

    # ------------------------------------------------------------------
    # Summary table (mean ± std per velocity, over surviving trials)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("SUMMARY (mean ± std over surviving trials)")
    print("=" * 70)
    header = (f"{'cmd_vx':>7} | {'survival':>8} | {'mean_vx':>14} | "
              f"{'track_err':>14} | {'drift':>14} | {'h_std':>10}")
    print(header)
    print("-" * len(header))

    for v in VELOCITIES:
        rows = [r for r in all_rows if r["cmd_vx"] == v]
        survived = [r for r in rows if not r["fell"]]
        n_surv = len(survived)
        surv_rate = n_surv / len(rows)

        if n_surv == 0:
            print(f"{v:>7.2f} | {surv_rate:>7.0%} | {'ALL FELL':>14} | "
                  f"{'-':>14} | {'-':>14} | {'-':>10}")
            continue

        def ms(key):
            vals = np.array([r[key] for r in survived], dtype=float)
            return vals.mean(), vals.std()

        vx_m, vx_s = ms("mean_vx")
        te_m, te_s = ms("vel_track_err")
        dr_m, dr_s = ms("lateral_drift")
        hs_m, hs_s = ms("height_std")

        print(f"{v:>7.2f} | {surv_rate:>7.0%} | "
              f"{vx_m:>6.3f}±{vx_s:>5.3f} | "
              f"{te_m:>6.3f}±{te_s:>5.3f} | "
              f"{dr_m:>6.3f}±{dr_s:>5.3f} | "
              f"{hs_m:>5.3f}±{hs_s:>4.3f}")

    print("=" * 70)
    print("Experiment 1 complete.")


if __name__ == "__main__":
    main()