"""
Regenerate paper Table II (velocity) and Table III (gait) mechanically from
the committed CSVs, with both population (ddof=0, matching exp1/exp2's own
np.std() calls) and sample (ddof=1) standard deviation, and print ready-to-
paste LaTeX tabular bodies plus a side-by-side diff against the values
currently printed in the manuscript.

Read-only: does not modify any CSV, does not re-run any trial.
"""

import os
import csv
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
EXP1_CSV = os.path.join(RESULTS_DIR, "exp1_velocity_sweep.csv")
EXP2_CSV = os.path.join(RESULTS_DIR, "exp2_gait_robustness.csv")


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_float(v):
    return None if v in ("", None) else float(v)


def group_metric(rows, group_key, group_val, metric):
    vals = [
        to_float(r[metric])
        for r in rows
        if r[group_key] == group_val and r["fell"] == "False"
    ]
    return np.array(vals, dtype=float)


def mean_std(arr, ddof):
    return float(arr.mean()), float(arr.std(ddof=ddof))


# ----------------------------------------------------------------------------
# Paper's printed values (Table II and Table III, transcribed verbatim from
# the manuscript for the side-by-side diff)
# ----------------------------------------------------------------------------
PAPER_TABLE_II = {
    # cmd: (achieved_mean, achieved_std, track_err, lateral_drift, height_std)
    0.00: (0.072, 0.003, 0.072, 0.043, 0.006),
    0.25: (0.143, 0.003, 0.108, 0.111, 0.005),
    0.50: (0.227, 0.005, 0.273, 0.162, 0.006),
    0.75: (0.368, 0.005, 0.383, 0.135, 0.007),
    1.00: (0.503, 0.004, 0.497, 0.018, 0.007),
    1.50: (0.557, 0.005, 0.943, 0.348, 0.006),
}

PAPER_TABLE_III = {
    # gait: (track_err_mean, track_err_std, drift_mean, drift_std, hstd_mean, hstd_std)
    "trot":  (0.273, 0.005, 0.162, 0.007, 0.006, 0.000),
    "pace":  (0.173, 0.004, 0.086, 0.008, 0.011, 0.001),
    "bound": (0.333, 0.007, 0.202, 0.004, 0.002, 0.000),
}

VELOCITIES = [0.00, 0.25, 0.50, 0.75, 1.00, 1.50]
GAITS = ["trot", "pace", "bound"]


def fmt3(x):
    return f"{x:.3f}"


def main():
    exp1_rows = load_rows(EXP1_CSV)
    exp2_rows = load_rows(EXP2_CSV)

    print("#" * 78)
    print("# TASK 2 — zero-variance check (raw per-seed values)")
    print("#" * 78)
    for label, rows, key, vals in (
        ("exp1", exp1_rows, "cmd_vx", [str(v) for v in VELOCITIES]),
        ("exp2", exp2_rows, "gait", GAITS),
    ):
        for v in vals:
            for metric in ("height_std", "mean_vx", "lateral_drift", "vel_track_err"):
                arr = group_metric(rows, key, v, metric)
                if arr.size == 0:
                    continue
                is_const = np.all(arr == arr[0])
                flag = " <-- ALL 5 SEEDS IDENTICAL (true std = 0)" if is_const else ""
                print(f"{label} {key}={v:>5} {metric:16s} raw={list(arr)}{flag}")
        print()

    print("#" * 78)
    print("# TASK 1 — Regenerated Table II (velocity sweep)")
    print("#" * 78)
    print(f"{'cmd':>5} | {'achieved(ddof0)':>18} | {'achieved(ddof1)':>18} | "
          f"{'track_err':>10} | {'drift':>10} | {'h_std':>10}")
    table_ii = {}
    for v in VELOCITIES:
        vx = group_metric(exp1_rows, "cmd_vx", str(v), "mean_vx")
        te = group_metric(exp1_rows, "cmd_vx", str(v), "vel_track_err")
        dr = group_metric(exp1_rows, "cmd_vx", str(v), "lateral_drift")
        hs = group_metric(exp1_rows, "cmd_vx", str(v), "height_std")

        vx_m0, vx_s0 = mean_std(vx, 0)
        vx_m1, vx_s1 = mean_std(vx, 1)
        te_m, _ = mean_std(te, 1)
        dr_m, _ = mean_std(dr, 1)
        hs_m, _ = mean_std(hs, 1)

        table_ii[v] = dict(vx_m0=vx_m0, vx_s0=vx_s0, vx_m1=vx_m1, vx_s1=vx_s1,
                            te_m=te_m, dr_m=dr_m, hs_m=hs_m)

        print(f"{v:>5.2f} | {vx_m0:.3f} ± {vx_s0:.3f}     | "
              f"{vx_m1:.3f} ± {vx_s1:.3f}     | "
              f"{te_m:>10.3f} | {dr_m:>10.3f} | {hs_m:>10.3f}")

    print()
    print("LaTeX tabular body — Table II, ddof=1 (sample std, RECOMMENDED):")
    print("-" * 70)
    for v in VELOCITIES:
        t = table_ii[v]
        print(f"{v:.2f} & {t['vx_m1']:.3f} $\\pm$ {t['vx_s1']:.3f} & "
              f"{t['te_m']:.3f} & {t['dr_m']:.3f} & {t['hs_m']:.3f} & 5/5 \\\\")

    print()
    print("LaTeX tabular body — Table II, ddof=0 (population std, current script convention):")
    print("-" * 70)
    for v in VELOCITIES:
        t = table_ii[v]
        print(f"{v:.2f} & {t['vx_m0']:.3f} $\\pm$ {t['vx_s0']:.3f} & "
              f"{t['te_m']:.3f} & {t['dr_m']:.3f} & {t['hs_m']:.3f} & 5/5 \\\\")

    print()
    print("#" * 78)
    print("# Side-by-side: Table II, paper vs. regenerated (ddof=1)")
    print("#" * 78)
    print(f"{'cmd':>5} {'field':>12} {'paper':>10} {'regen':>10} {'delta':>10}")
    for v in VELOCITIES:
        p = PAPER_TABLE_II[v]
        t = table_ii[v]
        rows = [
            ("achieved_mean", p[0], t["vx_m1"]),
            ("achieved_std", p[1], t["vx_s1"]),
            ("track_err", p[2], t["te_m"]),
            ("lat_drift", p[3], t["dr_m"]),
            ("height_std", p[4], t["hs_m"]),
        ]
        for field, pv, rv in rows:
            delta = rv - pv
            flag = "  <-- LARGE" if abs(delta) >= 0.002 else ""
            print(f"{v:>5.2f} {field:>12} {pv:>10.3f} {rv:>10.3f} {delta:>+10.3f}{flag}")

    print()
    print("#" * 78)
    print("# TASK 1 — Regenerated Table III (gait robustness)")
    print("#" * 78)
    table_iii = {}
    for g in GAITS:
        te = group_metric(exp2_rows, "gait", g, "vel_track_err")
        dr = group_metric(exp2_rows, "gait", g, "lateral_drift")
        hs = group_metric(exp2_rows, "gait", g, "height_std")

        te_m0, te_s0 = mean_std(te, 0)
        te_m1, te_s1 = mean_std(te, 1)
        dr_m0, dr_s0 = mean_std(dr, 0)
        dr_m1, dr_s1 = mean_std(dr, 1)
        hs_m0, hs_s0 = mean_std(hs, 0)
        hs_m1, hs_s1 = mean_std(hs, 1)

        table_iii[g] = dict(te_m0=te_m0, te_s0=te_s0, te_m1=te_m1, te_s1=te_s1,
                             dr_m0=dr_m0, dr_s0=dr_s0, dr_m1=dr_m1, dr_s1=dr_s1,
                             hs_m0=hs_m0, hs_s0=hs_s0, hs_m1=hs_m1, hs_s1=hs_s1)

        print(f"{g:>6} | te(d0)={te_m0:.3f}±{te_s0:.3f} te(d1)={te_m1:.3f}±{te_s1:.3f} | "
              f"dr(d0)={dr_m0:.3f}±{dr_s0:.3f} dr(d1)={dr_m1:.3f}±{dr_s1:.3f} | "
              f"hs(d0)={hs_m0:.3f}±{hs_s0:.3f} hs(d1)={hs_m1:.3f}±{hs_s1:.3f}")

    print()
    print("LaTeX tabular body — Table III, ddof=1 (sample std, RECOMMENDED):")
    print("-" * 70)
    for g in GAITS:
        t = table_iii[g]
        label = g.capitalize()
        print(f"{label} & {t['te_m1']:.3f} $\\pm$ {t['te_s1']:.3f} & "
              f"{t['dr_m1']:.3f} $\\pm$ {t['dr_s1']:.3f} & "
              f"{t['hs_m1']:.3f} $\\pm$ {t['hs_s1']:.3f} & 5/5 \\\\")

    print()
    print("LaTeX tabular body — Table III, ddof=0 (population std, current script convention):")
    print("-" * 70)
    for g in GAITS:
        t = table_iii[g]
        label = g.capitalize()
        print(f"{label} & {t['te_m0']:.3f} $\\pm$ {t['te_s0']:.3f} & "
              f"{t['dr_m0']:.3f} $\\pm$ {t['dr_s0']:.3f} & "
              f"{t['hs_m0']:.3f} $\\pm$ {t['hs_s0']:.3f} & 5/5 \\\\")

    print()
    print("#" * 78)
    print("# Side-by-side: Table III, paper vs. regenerated (ddof=1)")
    print("#" * 78)
    print(f"{'gait':>6} {'field':>10} {'paper':>10} {'regen':>10} {'delta':>10}")
    for g in GAITS:
        p = PAPER_TABLE_III[g]
        t = table_iii[g]
        rows = [
            ("track_err_m", p[0], t["te_m1"]),
            ("track_err_s", p[1], t["te_s1"]),
            ("drift_m", p[2], t["dr_m1"]),
            ("drift_s", p[3], t["dr_s1"]),
            ("hstd_m", p[4], t["hs_m1"]),
            ("hstd_s", p[5], t["hs_s1"]),
        ]
        for field, pv, rv in rows:
            delta = rv - pv
            flag = "  <-- LARGE" if abs(delta) >= 0.002 else ""
            print(f"{g:>6} {field:>10} {pv:>10.3f} {rv:>10.3f} {delta:>+10.3f}{flag}")


if __name__ == "__main__":
    main()
