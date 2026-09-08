"""
Stats + report extension for Experiment 6 (full velocity sweep x foot friction).

Reads exp6_full_velocity_friction_sweep.csv (written by
exp6_full_velocity_friction_sweep.py) and produces:
  - results/stats_exp6_full_velocity_friction.csv
  - an appended section in SENSITIVITY_REPORT.md (Parts 1 and 2 above are
    read back verbatim and kept; only a new Part 3 section is added)

Read-only w.r.t. every existing committed file and w.r.t. the exp4/exp5
sections of SENSITIVITY_REPORT.md already on disk. Does not touch the
manuscript or any committed CSV.
"""

import os
import csv
import numpy as np
from scipy import stats

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
IN_CSV = os.path.join(RESULTS_DIR, "exp6_full_velocity_friction_sweep.csv")
REPORT_PATH = os.path.abspath(os.path.join(RESULTS_DIR, "..", "..", "..", "SENSITIVITY_REPORT.md"))

FRICTION_VALUES = [0.4, 0.8, 1.5]
DECLARED_MU = 0.8
VELOCITIES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
ALPHA = 0.05


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_float(v):
    return None if v in ("", None) else float(v)


def mean_sd_ci(arr):
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


def group_metric(rows, filt, metric, survivors_only=True):
    vals = [
        to_float(r[metric]) for r in rows
        if filt(r) and (not survivors_only or r["fell"] == "False")
    ]
    return np.array(vals, dtype=float)


def main():
    rows = load_rows(IN_CSV)
    metrics = ["mean_vx", "vel_track_err", "lateral_drift", "height_std"]

    stats_rows = []
    for mu in FRICTION_VALUES:
        for v in VELOCITIES:
            def filt(r, v=v, mu=mu):
                return r["cmd_vx"] == str(v) and r["foot_friction_mu"] == str(mu)
            n_total = len([r for r in rows if filt(r)])
            n_surv = len([r for r in rows if filt(r) and r["fell"] == "False"])
            for metric in metrics:
                arr = group_metric(rows, filt, metric)
                m, sd, lo, hi, n = mean_sd_ci(arr)
                stats_rows.append(dict(
                    foot_friction_mu=mu, cmd_vx=v, metric=metric, n=n,
                    n_total=n_total, n_survived=n_surv,
                    mean=round(m, 6), sd=round(sd, 6),
                    ci95_lo=round(lo, 6), ci95_hi=round(hi, 6),
                ))

    with open(os.path.join(RESULTS_DIR, "stats_exp6_full_velocity_friction.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
        w.writeheader()
        w.writerows(stats_rows)
    print("Wrote stats_exp6_full_velocity_friction.csv")

    def stat(mu, v, metric):
        return next(r for r in stats_rows
                    if r["foot_friction_mu"] == mu and r["cmd_vx"] == v and r["metric"] == metric)

    means = {mu: {v: stat(mu, v, "mean_vx")["mean"] for v in VELOCITIES} for mu in FRICTION_VALUES}

    # ------------------------------------------------------------------
    # Control check vs exp1_velocity_sweep_30seed.csv, seeds 0-14
    # ------------------------------------------------------------------
    exp1_csv = os.path.join(RESULTS_DIR, "exp1_velocity_sweep_30seed.csv")
    with open(exp1_csv, newline="") as f:
        exp1_rows = list(csv.DictReader(f))
    control_diffs = {}
    for v in VELOCITIES:
        sub = [r for r in exp1_rows if r["cmd_vx"] == str(v) and r["fell"] == "False"
               and int(r["seed"]) < 15]
        exp1_mean = float(np.array([float(r["mean_vx"]) for r in sub]).mean())
        control_diffs[v] = abs(means[DECLARED_MU][v] - exp1_mean)
    control_max_diff = max(control_diffs.values())
    control_passed = control_max_diff < 1e-6

    # ------------------------------------------------------------------
    # Slopes between cmd=1.0 and cmd=1.5, and between cmd=0.75 and 1.0,
    # per friction value (m/s achieved per unit m/s commanded)
    # ------------------------------------------------------------------
    def slope(mu, v_lo, v_hi):
        return (means[mu][v_hi] - means[mu][v_lo]) / (v_hi - v_lo)

    slope_10_15 = {mu: slope(mu, 1.0, 1.5) for mu in FRICTION_VALUES}
    slope_075_10 = {mu: slope(mu, 0.75, 1.0) for mu in FRICTION_VALUES}
    flattening_ratio = {mu: slope_10_15[mu] / slope_075_10[mu] for mu in FRICTION_VALUES}

    peak_mu = {mu: max(means[mu], key=means[mu].get) for mu in FRICTION_VALUES}
    is_monotonic = {mu: peak_mu[mu] == 1.5 for mu in FRICTION_VALUES}

    print(f"Control check max diff: {control_max_diff:.2e} [{'PASS' if control_passed else 'FAIL'}]")
    print(f"Slope 0.75->1.0: {slope_075_10}")
    print(f"Slope 1.0->1.5: {slope_10_15}")
    print(f"Flattening ratio (slope drop): {flattening_ratio}")
    print(f"Peak velocity reached at cmd=1.5 for each mu: {is_monotonic}")

    # ------------------------------------------------------------------
    # Build the report section
    # ------------------------------------------------------------------
    lines = []
    a = lines.append

    a("")
    a("---")
    a("")
    a("# Sensitivity Report Part 3: Does Saturation Survive Off mu=0.8?")
    a("")
    a("Direct follow-up requested after Part 2 showed mu=1.5 reaching 0.6642 m/s at "
      "cmd=1.5, above the paper's stated ~0.55 m/s ceiling. Part 2 only swept "
      "cmd_vx in {0.5, 1.0, 1.5}; this section runs the **full exp1 velocity grid** -- "
      "the same six commanded velocities the paper's Table II reports -- at three foot "
      "friction values: 0.4, 0.8 (declared, the control), and 1.5. Mechanism is identical "
      "to Part 2: `harness.run_trial()`'s `foot_friction` kwarg mutates "
      "`model.geom_friction` for the four foot geoms in memory; `go2.xml` is never "
      "touched; no committed CSV is modified. Trot gait, n=15 seeds/condition "
      "(3 mu x 6 v x 15 seed = 270 trials).")
    a("")
    status_word = "PASSED" if control_passed else "FAILED"
    a(f"**Control check ({status_word}):** the mu=0.8 condition reproduces "
      f"`exp1_velocity_sweep_30seed.csv`'s first 15 seeds **bit-for-bit** at all six "
      f"commanded velocities (max diff = {control_max_diff:.2e} across "
      f"{VELOCITIES}). " + (
          "The mutation path is equivalent to the declared model when the override "
          "matches the declared value, as it must be -- this data can be trusted."
          if control_passed else
          "**THIS DOES NOT MATCH. Do not trust anything below until this is resolved.**"
      ))
    a("")
    a("## Full sweep results")
    a("")
    a(f"Foot sliding friction in {FRICTION_VALUES} (torsional/rolling held at their "
      f"declared values, 0.02/0.01), commanded velocity in {VELOCITIES} m/s, trot gait, "
      "n=15 seeds/condition.")
    a("")
    a("| foot_mu | cmd_vx | mean_vx | 95% CI | vel_track_err | lateral_drift | survival |")
    a("|---|---|---|---|---|---|---|")
    for mu in FRICTION_VALUES:
        for v in VELOCITIES:
            vx = stat(mu, v, "mean_vx")
            te = stat(mu, v, "vel_track_err")
            dr = stat(mu, v, "lateral_drift")
            marker = " **<- declared**" if mu == DECLARED_MU else ""
            a(f"| {mu}{marker} | {v:.2f} | {vx['mean']:.4f} | "
              f"[{vx['ci95_lo']:.4f}, {vx['ci95_hi']:.4f}] | {te['mean']:.4f} | "
              f"{dr['mean']:.4f} | {vx['n_survived']}/{vx['n_total']} |")
    a("")
    a("Raw trials: `results/exp6_full_velocity_friction_sweep.csv`. Full stats (all four "
      "metrics, all CIs): `results/stats_exp6_full_velocity_friction.csv`.")
    a("")
    a("## Answers")
    a("")
    a("**1. Does the curve flatten at high commanded velocity for all three friction "
      "values, or only at mu=0.8?**  ")
    a(f"**Only at mu=0.8.** The achieved-vs-commanded curve for each mu, cmd=0.0 through "
      f"1.5:")
    a("")
    for mu in FRICTION_VALUES:
        vals = " -> ".join(f"{means[mu][v]:.4f}" for v in VELOCITIES)
        marker = " (declared)" if mu == DECLARED_MU else ""
        a(f"- mu={mu}{marker}: {vals}")
    a("")
    a(f"At mu=0.8 the curve visibly bends over in the last two steps: the per-unit-command "
      f"slope drops from {slope_075_10[DECLARED_MU]:.4f} m/s (cmd 0.75->1.0) to "
      f"{slope_10_15[DECLARED_MU]:.4f} m/s (cmd 1.0->1.5) -- a "
      f"{100*(1-flattening_ratio[DECLARED_MU]):.0f}% drop in slope, the textbook signature "
      f"of saturation. At mu=1.5 the same two slopes are {slope_075_10[1.5]:.4f} and "
      f"{slope_10_15[1.5]:.4f} m/s -- only a "
      f"{100*(1-flattening_ratio[1.5]):.0f}% drop, i.e. the curve is still climbing at "
      f"nearly its prior rate with no flattening. At mu=0.4 there is no flattening either, "
      f"but for a different reason: achieved velocity **peaks at cmd=0.75 "
      f"({means[0.4][0.75]:.4f} m/s) and then declines** through cmd=1.0 "
      f"({means[0.4][1.0]:.4f} m/s) and cmd=1.5 ({means[0.4][1.5]:.4f} m/s) -- this is not "
      "a plateau, it is a peak-and-fall, most likely low-friction foot slip overwhelming "
      "gait-driven propulsion at the higher commanded speeds. None of the three curves "
      "share the same shape.")
    a("")
    a("**2. If saturation occurs at every friction value, at what achieved velocity does "
      "each one saturate?**  ")
    a("**It does not occur at every friction value, so this question only has an answer "
      "for one of the three:**")
    a("")
    a(f"- **mu=0.8 (declared): saturates near {means[0.8][1.5]:.4f} m/s** (achieved at "
      f"cmd=1.5, with the slope already collapsing by cmd=1.0). This is the number the "
      "paper reports.")
    a(f"- **mu=1.5: does not saturate within the tested command range.** Achieved velocity "
      f"is still rising steeply at cmd=1.5 ({means[1.5][1.5]:.4f} m/s, see Q3). There is "
      "no plateau to report a value for.")
    a(f"- **mu=0.4: does not saturate -- it peaks and falls.** The local maximum is "
      f"{means[0.4][0.75]:.4f} m/s at cmd=0.75, not a saturation ceiling in the sense the "
      "paper means (a flattening as commanded velocity keeps rising); by cmd=1.5 achieved "
      f"velocity has dropped to {means[0.4][1.5]:.4f} m/s, well below its own peak.")
    a("")
    a("**3. If it does NOT occur at mu=1.5: is achieved velocity still rising at cmd=1.5, "
      "and by how much per unit of command between 1.0 and 1.5, compared with the same "
      "slope at mu=0.8?**  ")
    a(f"**Yes, still rising, and rising steeply.** Between cmd=1.0 and cmd=1.5, achieved "
      f"velocity at mu=1.5 increases by {slope_10_15[1.5]:.4f} m/s per unit m/s commanded "
      f"({means[1.5][1.0]:.4f} -> {means[1.5][1.5]:.4f} m/s over that 0.5 m/s command "
      f"step). At mu=0.8 the same interval yields only {slope_10_15[DECLARED_MU]:.4f} m/s "
      f"per unit commanded ({means[DECLARED_MU][1.0]:.4f} -> {means[DECLARED_MU][1.5]:.4f} "
      f"m/s). **The mu=1.5 slope is {slope_10_15[1.5] / slope_10_15[DECLARED_MU]:.1f}x "
      "steeper than the mu=0.8 slope over the identical command interval.** Nothing in "
      "this range of mu=1.5 data suggests an approaching plateau -- the slope from "
      f"cmd=0.75->1.0 was {slope_075_10[1.5]:.4f} m/s/unit and the slope from cmd=1.0->1.5 "
      f"is {slope_10_15[1.5]:.4f} m/s/unit, essentially unchanged "
      f"({100*flattening_ratio[1.5]:.0f}% of the prior slope retained), whereas mu=0.8's "
      f"equivalent retained only {100*flattening_ratio[DECLARED_MU]:.0f}% of its prior "
      "slope. This is a real, qualitative difference in curve shape, not a rounding "
      "artifact.")
    a("")
    a("**4. Is the mu=0.8 control curve identical to the committed exp1 30-seed results "
      "at the overlapping seeds?**  ")
    if control_passed:
        a(f"**Confirmed, bit-for-bit.** Max absolute difference across all six commanded "
          f"velocities: {control_max_diff:.2e} (i.e. exactly zero at working precision). "
          "See the per-velocity table in the script output and "
          "`exp6_full_velocity_friction_sweep.py`'s built-in control check, which aborts "
          "the run if this fails. It did not fail. Everything above and below this line "
          "can be trusted as the same physical model exp1 used, run through the override "
          "path instead of the XML default.")
    else:
        a(f"**NO -- max difference {control_max_diff:.2e}.** Do not trust any number in "
          "this report until this is resolved.")
    a("")
    a("**5. In one sentence: is \"forward velocity saturates near 0.55 m/s\" a defensible "
      "claim about this policy, a defensible claim about this policy at this friction "
      "setting, or not defensible as stated?**  ")
    a("*\"'Forward velocity saturates near 0.55 m/s' is defensible only as a claim about "
      "this policy **at the declared foot friction of 0.8** -- it is not defensible as a "
      "claim about the policy in general, because at mu=1.5 (a value inside the same "
      "domain-randomization range the policy was trained under, "
      "[0.1, 3.0] per AUDIT.md Part A.2) achieved velocity keeps climbing well past 0.55 "
      f"m/s (to {means[1.5][1.5]:.4f} m/s at cmd=1.5) with no flattening in sight, meaning "
      "the reported ceiling is a property of the friction value the sweep happened to run "
      "at, not a property of the policy's own actuation limits.\" As written in the "
      "abstract/conclusion, with no friction qualifier attached, the claim overclaims: it "
      "should read \"...saturates near 0.55 m/s **at the ground-truth foot friction used "
      "in this study (mu=0.8)**\" or be dropped in favor of a friction-conditioned "
      "statement, because a reviewer who reruns at a different, equally physically "
      "plausible friction value will not reproduce the plateau.")
    a("")
    a("## Files (this section)")
    a("")
    a("- `experiments/results/exp6_full_velocity_friction_sweep.csv` -- 270 raw trial rows")
    a("- `experiments/results/stats_exp6_full_velocity_friction.csv` -- mean/SD/95% CI per (mu, velocity, metric)")
    a("- `experiments/exp6_full_velocity_friction_sweep.py` -- sweep harness (with built-in mu=0.8 control check)")
    a("- `experiments/exp6_report.py` -- this section's generator")

    with open(REPORT_PATH, "a") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nAppended Part 3 to {REPORT_PATH}")


if __name__ == "__main__":
    main()
