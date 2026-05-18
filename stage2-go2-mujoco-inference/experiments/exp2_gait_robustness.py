"""
Experiment 2 — Gait Robustness under sim-to-sim transfer.

Research question: walk-these-ways is gait-conditioned (trot/pace/bound
selectable via command). Does this conditioning survive transfer from
Isaac Gym to MuJoCo, or do non-trot gaits degrade/fail?

Protocol:
  - Terrain: flat
  - Commanded forward velocity: fixed 0.5 m/s
  - Gaits: {trot, pace, bound}
  - 5 independent trials per gait (seeds 0-4), 3s settle + 30s measure

Outputs:
  - results/exp2_gait_robustness.csv
  - Printed summary table (survival, tracking, drift per gait)
"""

import os
import csv
import numpy as np
import torch

from harness import run_trial, POLICY_DIR

SCENE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "scenes", "go2_flat.xml")
)
GAITS = ["trot", "pace", "bound"]
CMD_VX = 0.5
N_TRIALS = 5
SETTLE_S = 3.0
MEASURE_S = 30.0
RESULTS_CSV = os.path.join(os.path.dirname(__file__), "results",
                           "exp2_gait_robustness.csv")

CSV_FIELDS = [
    "scene", "cmd_vx", "cmd_vy", "cmd_yaw", "gait", "seed",
    "fell", "fall_time_s", "mean_vx", "mean_vy", "vel_track_err",
    "lateral_drift", "height_mean", "height_std", "distance_traveled",
]


def main():
    print("=" * 70)
    print("EXPERIMENT 2 — Gait Robustness (flat ground, cmd_vx=0.5 m/s)")
    print("=" * 70)
    print(f"Gaits: {GAITS}")
    print(f"Trials per gait: {N_TRIALS}")
    print(f"Total trials: {len(GAITS) * N_TRIALS}")
    print("=" * 70)

    print("Loading policy networks...")
    body_net = torch.jit.load(f"{POLICY_DIR}/body_latest.jit")
    body_net.eval()
    adapt_net = torch.jit.load(f"{POLICY_DIR}/adaptation_module_latest.jit")
    adapt_net.eval()
    print("Policy loaded.\n")

    all_rows = []

    for gait in GAITS:
        print(f"--- Gait: {gait} ---")
        for seed in range(N_TRIALS):
            result = run_trial(
                SCENE,
                lin_vel_x=CMD_VX, lin_vel_y=0.0, ang_vel_yaw=0.0,
                gait=gait,
                settle_s=SETTLE_S, measure_s=MEASURE_S,
                body_net=body_net, adapt_net=adapt_net,
                seed=seed,
            )
            all_rows.append(result)
            if result["fell"]:
                print(f"  seed {seed}: FELL @ {result['fall_time_s']:.1f}s")
            else:
                print(f"  seed {seed}: vx={result['mean_vx']:.3f} "
                      f"drift={result['lateral_drift']:.3f} "
                      f"h_std={result['height_std']:.3f}")
        print()

    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"Wrote {len(all_rows)} rows to {RESULTS_CSV}\n")

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    header = (f"{'gait':>6} | {'survival':>8} | {'mean_vx':>13} | "
              f"{'track_err':>13} | {'drift':>13} | {'h_std':>11}")
    print(header)
    print("-" * len(header))

    for gait in GAITS:
        rows = [r for r in all_rows if r["gait"] == gait]
        survived = [r for r in rows if not r["fell"]]
        surv_rate = len(survived) / len(rows)

        if not survived:
            print(f"{gait:>6} | {surv_rate:>7.0%} | {'ALL FELL':>13} | "
                  f"{'-':>13} | {'-':>13} | {'-':>11}")
            continue

        def ms(key):
            vals = np.array([r[key] for r in survived], dtype=float)
            return vals.mean(), vals.std()

        vx_m, vx_s = ms("mean_vx")
        te_m, te_s = ms("vel_track_err")
        dr_m, dr_s = ms("lateral_drift")
        hs_m, hs_s = ms("height_std")

        print(f"{gait:>6} | {surv_rate:>7.0%} | "
              f"{vx_m:>6.3f}±{vx_s:>5.3f} | "
              f"{te_m:>6.3f}±{te_s:>5.3f} | "
              f"{dr_m:>6.3f}±{dr_s:>5.3f} | "
              f"{hs_m:>5.3f}±{hs_s:>4.3f}")

    print("=" * 70)
    print("Experiment 2 complete.")


if __name__ == "__main__":
    main()