"""
Statistical analysis of the 30-seed re-run (Phase 2, Task 8).

Reads the *_30seed.csv files and produces:
  - results/stats_exp1_velocity.csv   (mean, sample SD, 95% CI per condition/metric)
  - results/stats_exp2_gait.csv       (mean, sample SD, 95% CI per gait/metric)
  - results/stats_exp2_pairwise.csv   (Welch's t-test + Holm + Cohen's d, trot/pace/bound)
  - results/stats_exp3_terrain.csv    (survival rate + Wilson score CI per condition)
  - STATS_REPORT.md                   (readable summary)

Read-only w.r.t. the 30seed CSVs; does not touch the 5-seed files.
"""

import os
import csv
import itertools
import numpy as np
from scipy import stats

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
EXP1_CSV = os.path.join(RESULTS_DIR, "exp1_velocity_sweep_30seed.csv")
EXP2_CSV = os.path.join(RESULTS_DIR, "exp2_gait_robustness_30seed.csv")
EXP3_CSV = os.path.join(RESULTS_DIR, "exp3_terrain_30seed.csv")

VELOCITIES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
GAITS = ["trot", "pace", "bound"]
SLOPE_ANGLES = [5, 10, 15, 20, 25]
STEP_HEIGHTS_CM = [2, 5, 8, 12, 16]
N_TRIALS = 30
ALPHA = 0.05


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_float(v):
    return None if v in ("", None) else float(v)


def group_metric(rows, group_key, group_val, metric, survivors_only=True):
    vals = [
        to_float(r[metric])
        for r in rows
        if r[group_key] == group_val and (not survivors_only or r["fell"] == "False")
    ]
    return np.array(vals, dtype=float)


def mean_sd_ci(arr):
    """Sample mean, sample SD (ddof=1), and 95% CI via t-distribution."""
    n = arr.size
    m = float(arr.mean())
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    if n > 1 and sd > 0:
        se = sd / np.sqrt(n)
        tcrit = stats.t.ppf(1 - ALPHA / 2, df=n - 1)
        half = tcrit * se
    else:
        half = 0.0
    return m, sd, m - half, m + half, n


def wilson_ci(successes, n, z=1.959963984540054):
    """Wilson score interval for a binomial proportion (95% by default)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def cohens_d_welch(a, b):
    """Cohen's d using the pooled SD (standard definition); reported alongside
    Welch's t-test, which itself does not assume equal variances."""
    na, nb = a.size, b.size
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled_sd = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled_sd)


def holm_correction(pvals):
    """Holm-Bonferroni step-down correction. Returns adjusted p-values in the
    original order of `pvals`."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running_max = max(running_max, val)
        adj[idx] = min(1.0, running_max)
    return adj


def main():
    exp1_rows = load_rows(EXP1_CSV)
    exp2_rows = load_rows(EXP2_CSV)
    exp3_rows = load_rows(EXP3_CSV)

    metrics = ["mean_vx", "vel_track_err", "lateral_drift", "height_std"]

    # ------------------------------------------------------------------
    # exp1: per-velocity stats
    # ------------------------------------------------------------------
    exp1_stats_rows = []
    for v in VELOCITIES:
        n_total = len([r for r in exp1_rows if r["cmd_vx"] == str(v)])
        n_surv = len([r for r in exp1_rows if r["cmd_vx"] == str(v) and r["fell"] == "False"])
        for metric in metrics:
            arr = group_metric(exp1_rows, "cmd_vx", str(v), metric)
            m, sd, lo, hi, n = mean_sd_ci(arr)
            exp1_stats_rows.append(dict(
                cmd_vx=v, metric=metric, n=n, n_total=n_total, n_survived=n_surv,
                mean=round(m, 5), sd=round(sd, 5),
                ci95_lo=round(lo, 5), ci95_hi=round(hi, 5),
            ))

    with open(os.path.join(RESULTS_DIR, "stats_exp1_velocity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(exp1_stats_rows[0].keys()))
        w.writeheader()
        w.writerows(exp1_stats_rows)

    # ------------------------------------------------------------------
    # exp2: per-gait stats
    # ------------------------------------------------------------------
    exp2_stats_rows = []
    gait_arrays = {g: {} for g in GAITS}
    for g in GAITS:
        for metric in metrics:
            arr = group_metric(exp2_rows, "gait", g, metric)
            gait_arrays[g][metric] = arr
            m, sd, lo, hi, n = mean_sd_ci(arr)
            exp2_stats_rows.append(dict(
                gait=g, metric=metric, n=n,
                mean=round(m, 5), sd=round(sd, 5),
                ci95_lo=round(lo, 5), ci95_hi=round(hi, 5),
            ))

    with open(os.path.join(RESULTS_DIR, "stats_exp2_gait.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(exp2_stats_rows[0].keys()))
        w.writeheader()
        w.writerows(exp2_stats_rows)

    # ------------------------------------------------------------------
    # exp2: pairwise Welch's t-tests + Holm correction + Cohen's d
    # (tracking error, lateral drift, height std -- the paper's Table III metrics)
    # ------------------------------------------------------------------
    pair_metrics = ["vel_track_err", "lateral_drift", "height_std"]
    pairs = list(itertools.combinations(GAITS, 2))

    raw_records = []
    for metric in pair_metrics:
        for g1, g2 in pairs:
            a = gait_arrays[g1][metric]
            b = gait_arrays[g2][metric]
            tstat, pval = stats.ttest_ind(a, b, equal_var=False)
            d = cohens_d_welch(a, b)
            raw_records.append(dict(
                metric=metric, gait_a=g1, gait_b=g2,
                mean_a=round(float(a.mean()), 5), mean_b=round(float(b.mean()), 5),
                t_stat=round(float(tstat), 4), p_raw=float(pval),
                cohens_d=round(d, 4),
            ))

    # Holm correction applied across the whole family of 9 comparisons
    # (3 metrics x 3 gait pairs), as specified.
    pvals = np.array([r["p_raw"] for r in raw_records])
    p_holm = holm_correction(pvals)
    for r, p_adj in zip(raw_records, p_holm):
        # Full precision preserved (these underflow to 0.0 if rounded to a
        # fixed number of decimals) -- use scientific notation instead.
        r["p_raw"] = f"{r['p_raw']:.4e}"
        r["p_holm"] = f"{float(p_adj):.4e}"
        r["significant_holm_0.05"] = bool(p_adj < ALPHA)

    with open(os.path.join(RESULTS_DIR, "stats_exp2_pairwise.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(raw_records[0].keys()))
        w.writeheader()
        w.writerows(raw_records)

    # ------------------------------------------------------------------
    # exp3: survival rate + Wilson CI, and progress-threshold traversability
    # ------------------------------------------------------------------
    PROGRESS_THRESHOLD_M = 3.5
    exp3_stats_rows = []

    def terrain_condition_stats(label, scene_file):
        rows = [r for r in exp3_rows if r["scene"] == scene_file]
        n = len(rows)
        survived = [r for r in rows if r["fell"] == "False"]
        n_surv = len(survived)
        p, lo, hi = wilson_ci(n_surv, n)
        if survived:
            dists = np.array([to_float(r["distance_traveled"]) for r in survived])
            vxs = np.array([to_float(r["mean_vx"]) for r in survived])
            dist_m, dist_sd = float(dists.mean()), float(dists.std(ddof=1)) if n_surv > 1 else 0.0
            vx_m, vx_sd = float(vxs.mean()), float(vxs.std(ddof=1)) if n_surv > 1 else 0.0
        else:
            dist_m = dist_sd = vx_m = vx_sd = None
        made_progress = (dist_m is not None) and (dist_m > PROGRESS_THRESHOLD_M)
        traversable = (p >= 0.8) and made_progress
        exp3_stats_rows.append(dict(
            condition=label, scene=scene_file, n=n, n_survived=n_surv,
            survival_rate=round(p, 4),
            survival_ci95_lo=round(lo, 4), survival_ci95_hi=round(hi, 4),
            mean_dist=round(dist_m, 4) if dist_m is not None else "",
            dist_sd=round(dist_sd, 4) if dist_m is not None else "",
            mean_vx=round(vx_m, 4) if vx_m is not None else "",
            vx_sd=round(vx_sd, 4) if vx_m is not None else "",
            traversable="YES" if traversable else "no",
        ))

    for a in SLOPE_ANGLES:
        terrain_condition_stats(f"slope_{a}deg", f"go2_slope_{a:02d}.xml")
    for h in STEP_HEIGHTS_CM:
        terrain_condition_stats(f"stairs_{h}cm", f"go2_stairs_{h:02d}.xml")

    with open(os.path.join(RESULTS_DIR, "stats_exp3_terrain.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(exp3_stats_rows[0].keys()))
        w.writeheader()
        w.writerows(exp3_stats_rows)

    print("Wrote stats_exp1_velocity.csv, stats_exp2_gait.csv, "
          "stats_exp2_pairwise.csv, stats_exp3_terrain.csv")

    return exp1_stats_rows, exp2_stats_rows, raw_records, exp3_stats_rows


if __name__ == "__main__":
    main()
