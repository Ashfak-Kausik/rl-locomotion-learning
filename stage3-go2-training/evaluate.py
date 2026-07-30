"""
Evaluate an exported Stage 3 policy against the walk-these-ways baseline.

Uses Stage 2's `harness.run_trial()` **unmodified** — same scenes, same seeded
initial conditions, same fall criteria, same metrics. That is what makes the
comparison meaningful rather than decorative: any difference in the numbers is
a difference in the policy, not in the measurement.

Baseline figures are read from the committed Experiment 1/3 CSVs, so the
comparison is against real recorded data rather than remembered numbers.

Usage
-----
    export GO2_POLICY_DIR=policies/stage3-my_run
    python evaluate.py                       # velocity sweep vs baseline
    python evaluate.py --terrain             # + terrain, the Stage 3 goal
    python evaluate.py --trials 5            # more seeds (default 3)
"""

import argparse
import csv
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE2 = REPO_ROOT / "stage2-go2-mujoco-inference"
for p in (str(STAGE2), str(STAGE2 / "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

import paths  # noqa: E402
from harness import run_trial  # noqa: E402

RESULTS = STAGE2 / "experiments" / "results"
SCENES = STAGE2 / "scenes"

# Terrain levels the flat-trained baseline could NOT clear (Experiment 3).
# Beating these is the headline claim a Stage 3 policy would be making.
BASELINE_TERRAIN = {
    "go2_slope_10.xml": ("10 deg slope", "100% survival, traversable"),
    "go2_slope_15.xml": ("15 deg slope", "20% survival, NOT traversable"),
    "go2_slope_20.xml": ("20 deg slope", "0% survival"),
    "go2_stairs_02.xml": ("2 cm steps", "traversable, marginal"),
    "go2_stairs_05.xml": ("5 cm steps", "SAFE STALL — 0.47 m, never climbs"),
}


def baseline_velocity_table():
    """Mean achieved vx per commanded vx, from the committed Exp 1 CSV."""
    path = RESULTS / "exp1_velocity_sweep.csv"
    if not path.is_file():
        return {}
    by_cmd = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["fell"] == "False" and row["mean_vx"] not in ("", "None"):
                by_cmd[float(row["cmd_vx"])].append(float(row["mean_vx"]))
    return {c: statistics.fmean(v) for c, v in by_cmd.items()}


def summarise(results):
    """Aggregate a list of trial dicts into (survival, mean_vx, mean_dist)."""
    survivors = [r for r in results if not r["fell"]]
    survival = len(survivors) / len(results) if results else 0.0
    if not survivors:
        return survival, None, None
    return (
        survival,
        statistics.fmean(r["mean_vx"] for r in survivors),
        statistics.fmean(r["distance_traveled"] for r in survivors),
    )


def eval_velocity(velocities, trials, body_net, adapt_net):
    print("=" * 78)
    print("VELOCITY TRACKING — Stage 3 policy vs walk-these-ways baseline")
    print("=" * 78)
    baseline = baseline_velocity_table()

    header = (f"{'cmd_vx':>7} | {'survival':>8} | {'achieved':>9} | "
              f"{'baseline':>9} | {'delta':>8} | {'verdict':>8}")
    print(header)
    print("-" * len(header))

    scene = str((SCENES / "go2_flat.xml").resolve())
    rows = []
    for cmd in velocities:
        results = [
            run_trial(scene, lin_vel_x=cmd, gait="trot",
                      settle_s=2.0, measure_s=10.0,
                      body_net=body_net, adapt_net=adapt_net, seed=s)
            for s in range(trials)
        ]
        survival, mean_vx, _ = summarise(results)
        base = baseline.get(cmd)

        if mean_vx is None:
            print(f"{cmd:>7.2f} | {survival:>7.0%} | {'ALL FELL':>9} | "
                  f"{base if base is None else f'{base:>9.3f}'} | "
                  f"{'-':>8} | {'-':>8}")
            rows.append((cmd, survival, None, base))
            continue

        if base is None:
            delta_s, verdict = "-", "-"
        else:
            delta = mean_vx - base
            delta_s = f"{delta:+.3f}"
            verdict = "better" if delta > 0.01 else (
                "worse" if delta < -0.01 else "same")

        print(f"{cmd:>7.2f} | {survival:>7.0%} | {mean_vx:>9.3f} | "
              f"{'-' if base is None else f'{base:.3f}':>9} | "
              f"{delta_s:>8} | {verdict:>8}")
        rows.append((cmd, survival, mean_vx, base))

    print("=" * 78)
    return rows


def eval_terrain(trials, body_net, adapt_net):
    print()
    print("=" * 78)
    print("TERRAIN — the Stage 3 objective")
    print("=" * 78)
    print("Baseline (Exp 3) traverses 10 deg slopes and 2 cm steps, and")
    print("safe-stalls at 5 cm. Anything past that is new capability.")
    print()

    header = (f"{'terrain':>14} | {'survival':>8} | {'distance':>9} | "
              f"{'baseline (Exp 3)':>32}")
    print(header)
    print("-" * len(header))

    for scene_file, (label, base_note) in BASELINE_TERRAIN.items():
        scene = str((SCENES / scene_file).resolve())
        if not os.path.isfile(scene):
            continue
        results = [
            run_trial(scene, lin_vel_x=0.5, gait="trot",
                      settle_s=2.0, measure_s=15.0,
                      body_net=body_net, adapt_net=adapt_net, seed=s)
            for s in range(trials)
        ]
        survival, _, mean_dist = summarise(results)
        dist_s = "-" if mean_dist is None else f"{mean_dist:.2f} m"
        print(f"{label:>14} | {survival:>7.0%} | {dist_s:>9} | {base_note:>32}")

    print("=" * 78)
    print("Reminder: survival alone is NOT traversal. Experiment 3 found the")
    print("baseline scoring 100% survival on 16 cm stairs while standing")
    print("motionless at the bottom. Distance is the honest metric.")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=3,
                    help="seeds per condition (default 3; experiments use 5)")
    ap.add_argument("--terrain", action="store_true",
                    help="also evaluate on slopes and stairs")
    ap.add_argument("--velocities", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
    args = ap.parse_args()

    print(f"policy: {paths.POLICY_DIR}")
    if not paths.policy_available():
        paths.require_policy()

    body_net, adapt_net = paths.load_policy()
    print(f"trials per condition: {args.trials}\n")

    eval_velocity(args.velocities, args.trials, body_net, adapt_net)
    if args.terrain:
        eval_terrain(args.trials, body_net, adapt_net)


if __name__ == "__main__":
    main()
