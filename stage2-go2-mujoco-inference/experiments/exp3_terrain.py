"""
Experiment 3 — Terrain Robustness (out-of-distribution generalization).

Research question: the walk-these-ways policy was trained on flat ground
only (terrain_proportions = flat). How far does a flat-trained policy
generalize to terrain it never saw before failing?

Protocol:
  3a Slopes: incline {5,10,15,20,25} deg, trot, cmd_vx=0.5 m/s, uphill
  3b Stairs: step height {2,5,8,12,16} cm, trot, cmd_vx=0.5 m/s
  5 independent trials per condition, 3s settle + 20s measure
  (20s, shorter than Exp1/2's 30s: terrain trials either succeed or fail
   fast; 20s is enough to traverse the run-up and engage the terrain)

Metrics per condition:
  - survival rate (fraction not falling)
  - mean forward progress (distance traveled, indicates ascent vs stall)
  - of survivors: tracking error, drift, height stability

"Traversable" threshold = survival >= 80% (>=4/5 trials).

Outputs:
  - results/exp3_terrain.csv
  - Printed summary tables (slopes, stairs)
"""

import os
import csv
import numpy as np
import torch

from harness import run_trial, POLICY_DIR

SCENES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "scenes")
)
SLOPE_ANGLES = [5, 10, 15, 20, 25]
STEP_HEIGHTS_CM = [2, 5, 8, 12, 16]
CMD_VX = 0.5
N_TRIALS = 5
SETTLE_S = 3.0
MEASURE_S = 20.0
RESULTS_CSV = os.path.join(os.path.dirname(__file__), "results",
                           "exp3_terrain.csv")

CSV_FIELDS = [
    "scene", "cmd_vx", "cmd_vy", "cmd_yaw", "gait", "seed",
    "fell", "fall_time_s", "mean_vx", "mean_vy", "vel_track_err",
    "lateral_drift", "height_mean", "height_std", "distance_traveled",
]


def run_condition(label, scene_file, body_net, adapt_net, all_rows):
    scene_path = os.path.join(SCENES_DIR, scene_file)
    print(f"--- {label}  ({scene_file}) ---")
    for seed in range(N_TRIALS):
        result = run_trial(
            scene_path,
            lin_vel_x=CMD_VX, lin_vel_y=0.0, ang_vel_yaw=0.0,
            gait="trot",
            settle_s=SETTLE_S, measure_s=MEASURE_S,
            body_net=body_net, adapt_net=adapt_net,
            seed=seed,
        )
        all_rows.append(result)
        if result["fell"]:
            print(f"  seed {seed}: FELL @ {result['fall_time_s']:.1f}s")
        else:
            print(f"  seed {seed}: SURVIVED  dist={result['distance_traveled']:.2f}m "
                  f"vx={result['mean_vx']:.3f}")
    print()


def summarize(label, conditions, all_rows, key_name):
    print("=" * 70)
    print(f"SUMMARY — {label}")
    print("=" * 70)
    header = (f"{key_name:>10} | {'survival':>8} | {'mean_dist':>11} | "
              f"{'mean_vx':>13} | {'traversable':>11}")
    print(header)
    print("-" * len(header))
    for cond_val, scene_file in conditions:
        rows = [r for r in all_rows if r["scene"] == scene_file]
        survived = [r for r in rows if not r["fell"]]
        surv_rate = len(survived) / len(rows) if rows else 0.0

        # Traversable requires BOTH survival AND genuine forward progress.
        # Run-up is 2.5 m; demand >3.5 m mean distance (>=1 m onto terrain)
        # so a robot that survives by stalling at the base is NOT counted
        # as traversing.
        PROGRESS_THRESHOLD_M = 3.5
        if survived:
            mean_dist_val = np.mean([r["distance_traveled"] for r in survived])
        else:
            mean_dist_val = 0.0
        made_progress = mean_dist_val > PROGRESS_THRESHOLD_M
        traversable = "YES" if (surv_rate >= 0.8 and made_progress) else "no"

        if survived:
            dists = np.array([r["distance_traveled"] for r in survived], float)
            vxs = np.array([r["mean_vx"] for r in survived], float)
            dist_str = f"{dists.mean():>5.2f}±{dists.std():>4.2f}"
            vx_str = f"{vxs.mean():>6.3f}±{vxs.std():>5.3f}"
        else:
            dist_str = f"{'-':>10}"
            vx_str = f"{'-':>13}"

        print(f"{cond_val:>10} | {surv_rate:>7.0%} | {dist_str:>11} | "
              f"{vx_str:>13} | {traversable:>11}")
    print("=" * 70)
    print()


def main():
    print("=" * 70)
    print("EXPERIMENT 3 — Terrain Robustness (flat-trained policy, OOD)")
    print("=" * 70)
    print(f"Slopes: {SLOPE_ANGLES} deg | Stairs: {STEP_HEIGHTS_CM} cm")
    print(f"cmd_vx={CMD_VX} m/s, trot, {N_TRIALS} trials each")
    print(f"Total trials: {(len(SLOPE_ANGLES)+len(STEP_HEIGHTS_CM))*N_TRIALS}")
    print("=" * 70)

    print("Loading policy networks...")
    body_net = torch.jit.load(f"{POLICY_DIR}/body_latest.jit")
    body_net.eval()
    adapt_net = torch.jit.load(f"{POLICY_DIR}/adaptation_module_latest.jit")
    adapt_net.eval()
    print("Policy loaded.\n")

    all_rows = []

    print(">>> 3a — SLOPES\n")
    for ang in SLOPE_ANGLES:
        run_condition(f"Slope {ang} deg", f"go2_slope_{ang:02d}.xml",
                       body_net, adapt_net, all_rows)

    print(">>> 3b — STAIRS\n")
    for h in STEP_HEIGHTS_CM:
        run_condition(f"Stairs {h} cm", f"go2_stairs_{h:02d}.xml",
                       body_net, adapt_net, all_rows)

    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"Wrote {len(all_rows)} rows to {RESULTS_CSV}\n")

    summarize("3a SLOPES", [(f"{a} deg", f"go2_slope_{a:02d}.xml")
                            for a in SLOPE_ANGLES], all_rows, "slope")
    summarize("3b STAIRS", [(f"{h} cm", f"go2_stairs_{h:02d}.xml")
                            for h in STEP_HEIGHTS_CM], all_rows, "step_h")

    print("Experiment 3 complete.")


if __name__ == "__main__":
    main()