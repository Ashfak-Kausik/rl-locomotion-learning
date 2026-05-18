"""
Generate paper figures from experiment CSVs.

Reads:
  results/exp1_velocity_sweep.csv
  results/exp2_gait_robustness.csv
  results/exp3_terrain.csv

Writes (300 dpi, paper-ready) to figures/:
  fig1_velocity_tracking.png
  fig2_lateral_drift.png
  fig3_gait_tradeoff.png
  fig4_terrain_robustness.png
"""

import os
import csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "results")
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})


def load_csv(name):
    path = os.path.join(RESULTS, name)
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(x):
    """Parse CSV cell to float or None."""
    if x is None or x == "" or x == "None":
        return None
    return float(x)


# ----------------------------------------------------------------------------
# Figure 1 — Achieved vs commanded velocity (the sim-to-sim gap)
# ----------------------------------------------------------------------------
def fig1():
    rows = load_csv("exp1_velocity_sweep.csv")
    by_cmd = defaultdict(list)
    for r in rows:
        if r["fell"] == "False":
            by_cmd[float(r["cmd_vx"])].append(fnum(r["mean_vx"]))

    cmds = sorted(by_cmd)
    means = [np.mean(by_cmd[c]) for c in cmds]
    stds = [np.std(by_cmd[c]) for c in cmds]

    fig, ax = plt.subplots(figsize=(6, 5))
    lim = max(max(cmds), max(means)) * 1.1
    ax.plot([0, lim], [0, lim], "--", color="gray",
            label="Ideal (perfect tracking)")
    ax.errorbar(cmds, means, yerr=stds, marker="o", capsize=4,
                color="C0", label="Achieved (MuJoCo)")
    ax.set_xlabel("Commanded forward velocity (m/s)")
    ax.set_ylabel("Achieved forward velocity (m/s)")
    ax.set_title("Sim-to-Sim Velocity Tracking Gap")
    ax.legend()
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    fig.savefig(os.path.join(FIGS, "fig1_velocity_tracking.png"))
    plt.close(fig)
    print("fig1_velocity_tracking.png")


# ----------------------------------------------------------------------------
# Figure 2 — Lateral drift vs commanded velocity (the cmd=1.0 sweet spot)
# ----------------------------------------------------------------------------
def fig2():
    rows = load_csv("exp1_velocity_sweep.csv")
    by_cmd = defaultdict(list)
    for r in rows:
        if r["fell"] == "False":
            by_cmd[float(r["cmd_vx"])].append(fnum(r["lateral_drift"]))

    cmds = sorted(by_cmd)
    means = [np.mean(by_cmd[c]) for c in cmds]
    stds = [np.std(by_cmd[c]) for c in cmds]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(cmds, means, yerr=stds, marker="s", capsize=4, color="C3")
    imin = int(np.argmin(means))
    ax.annotate(f"minimum drift\n@ {cmds[imin]:.2f} m/s",
                xy=(cmds[imin], means[imin]),
                xytext=(cmds[imin], means[imin] + 0.12),
                ha="center",
                arrowprops=dict(arrowstyle="->", color="black"))
    ax.set_xlabel("Commanded forward velocity (m/s)")
    ax.set_ylabel("Mean lateral drift |v_y| (m/s)")
    ax.set_title("Directional Stability vs. Commanded Velocity")
    fig.savefig(os.path.join(FIGS, "fig2_lateral_drift.png"))
    plt.close(fig)
    print("fig2_lateral_drift.png")


# ----------------------------------------------------------------------------
# Figure 3 — Gait tradeoff (grouped bars)
# ----------------------------------------------------------------------------
def fig3():
    rows = load_csv("exp2_gait_robustness.csv")
    gaits = ["trot", "pace", "bound"]
    metrics = ["vel_track_err", "lateral_drift", "height_std"]
    labels = ["Tracking error\n(m/s)", "Lateral drift\n(m/s)",
              "Height std\n(m)"]

    data = {g: {m: [] for m in metrics} for g in gaits}
    for r in rows:
        if r["fell"] == "False":
            g = r["gait"]
            for m in metrics:
                data[g][m].append(fnum(r[m]))

    means = {g: [np.mean(data[g][m]) for m in metrics] for g in gaits}
    stds = {g: [np.std(data[g][m]) for m in metrics] for g in gaits}

    x = np.arange(len(metrics))
    w = 0.25
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, g in enumerate(gaits):
        ax.bar(x + (i - 1) * w, means[g], w, yerr=stds[g], capsize=3,
               label=g)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Value (lower is better)")
    ax.set_title("Gait Performance Tradeoff (flat, cmd=0.5 m/s)")
    ax.legend(title="Gait")
    fig.savefig(os.path.join(FIGS, "fig3_gait_tradeoff.png"))
    plt.close(fig)
    print("fig3_gait_tradeoff.png")


# ----------------------------------------------------------------------------
# Figure 4 — Terrain robustness (two panels)
# ----------------------------------------------------------------------------
def fig4():
    rows = load_csv("exp3_terrain.csv")

    slope_rows = [r for r in rows if "slope" in r["scene"]]
    stair_rows = [r for r in rows if "stairs" in r["scene"]]

    def angle_of(s):  # go2_slope_15.xml -> 15
        return int(s.replace("go2_slope_", "").replace(".xml", ""))

    def step_of(s):   # go2_stairs_08.xml -> 8
        return int(s.replace("go2_stairs_", "").replace(".xml", ""))

    # Slopes: survival rate
    sl = defaultdict(list)
    for r in slope_rows:
        sl[angle_of(r["scene"])].append(r["fell"] == "False")
    angles = sorted(sl)
    surv = [100.0 * np.mean(sl[a]) for a in angles]

    # Stairs: mean distance (the stall plateau)
    st = defaultdict(list)
    for r in stair_rows:
        if r["fell"] == "False":
            st[step_of(r["scene"])].append(fnum(r["distance_traveled"]))
    steps = sorted(st)
    dist_m = [np.mean(st[s]) for s in steps]
    dist_s = [np.std(st[s]) for s in steps]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))

    a1.plot(angles, surv, marker="o", color="C2")
    a1.axhline(80, ls="--", color="gray", label="80% traversable threshold")
    a1.set_xlabel("Slope angle (degrees)")
    a1.set_ylabel("Survival rate (%)")
    a1.set_title("(a) Slope Robustness")
    a1.set_ylim(-5, 105)
    a1.legend()

    a2.errorbar(steps, dist_m, yerr=dist_s, marker="s", capsize=4,
                color="C1")
    a2.axhline(3.5, ls="--", color="gray",
               label="progress threshold (3.5 m)")
    a2.axhline(2.5, ls=":", color="darkred", label="run-up length (2.5 m)")
    a2.set_xlabel("Step height (cm)")
    a2.set_ylabel("Mean distance traveled (m)")
    a2.set_title("(b) Stair Robustness (stall plateau)")
    a2.legend()

    fig.suptitle("Out-of-Distribution Terrain Robustness "
                 "(flat-trained policy)")
    fig.savefig(os.path.join(FIGS, "fig4_terrain_robustness.png"))
    plt.close(fig)
    print("fig4_terrain_robustness.png")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    print(f"\nAll figures written to {FIGS}/")