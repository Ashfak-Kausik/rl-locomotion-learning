# Reproducibility Audit — `rl-locomotion-learning`

**Scope note (read first):** The manuscript itself (`Sim_to_Sim_Transfer_of_Reinforcement_Learning_Based_Quadruped_Locomotion...pdf`) is **not in this git repository**. It was located in `~/Downloads/` on this machine. All "paper" claims below are quoted from that PDF. All source-code claims are quoted from `github.com/Ashfak-Kausik/rl-locomotion-learning` (branch `main`, HEAD `17fbaf6`), which was already present locally and up to date with `origin` — no clone was necessary.

**Second scope note:** The pretrained checkpoint the whole pipeline depends on is loaded from a hardcoded absolute path that points **outside this repository**, into a sibling project `robot-dog-sim/walk-these-ways-go2` (also present on this machine, also pushed to GitHub as `Ashfak-Kausik/robot-dog-sim`). That sibling repo is *not* what you asked me to audit, but since it's the only place the training configuration actually exists, I read it read-only to answer Part A.2 with facts instead of "not determinable." Every fact sourced from it is labeled **[external repo]** below so you know it's not something a reviewer cloning `rl-locomotion-learning` alone could ever recover.

No files were modified, no experiments were re-run, nothing was committed.

---

## PART A — Reproducibility Facts

### A.1 — Where the pretrained policy comes from

- **Not vendored, not downloaded, not git-ignored — it simply isn't in this repo.** `harness.py:19-22` (and duplicated identically in `06_run_policy.py:27`, `05_inspect_policy.py:9`, `07_run_policy_interactive.py:36-39`) hardcodes:
  ```
  POLICY_DIR = "/home/user/projects/robot-dog-sim/walk-these-ways-go2/runs/gait-conditioned-agility/pretrain-go2/train/142238.667503/checkpoints"
  ```
  This is an absolute path unique to the author's machine. It is not a git submodule, not referenced by URL, not fetched by any script. A reviewer who clones `rl-locomotion-learning` gets code that immediately fails with `FileNotFoundError` — there is no checkpoint anywhere in the checked-out tree, and no instruction anywhere (README, script, comment) telling them where to obtain one.
- Checkpoint files (found only via the hardcoded path, **outside** the audited repo):
  | File | SHA-256 | Size |
  |---|---|---|
  | `body_latest.jit` | `7b6e604e2147742a89ef50d91e7ee501023331b2589d1c3143a9d2ba858db7b5` | 4,981,032 bytes |
  | `adaptation_module_latest.jit` | `0e091f829dcfbedd4ccca6752863e1e2feca105f79da07d04e3545b8815dcc13` | 2,293,757 bytes |
- **[external repo]** These files are *also* git-ignored in `robot-dog-sim/walk-these-ways-go2/.gitignore:12`: `/runs/gait-conditioned-agility/pretrain-go2/`. So the checkpoint is excluded from version control in **both** repositories. It exists only as local files on this one machine.
- **[external repo]** `robot-dog-sim/walk-these-ways-go2` is not the original MIT Improbable-AI `walk-these-ways` repo the paper cites (ref [9], Margolis & Agrawal). Its own `README.md:5` states it is "forked from [walk-these-ways](https://github.com/Improbable-AI/walk-these-ways)" via an intermediate third-party fork (`Teddy-Liao/walk-these-ways-go2`, cloned per that README's own instructions), then further adapted for Go2 and retrained by you. The paper (Section III.B) says only "trained externally in a GPU-based simulator" and cites [9] — it never discloses that the checkpoint is a self-trained Go2 policy from a community fork, not an official release artifact. Reviewers asking for training provenance are entitled to this chain; right now it's undiscoverable from either the repo or the paper text.
- **Verdict: NOT DETERMINABLE from `rl-locomotion-learning` alone.** The facts above required inspecting a second, unaudited repository that happens to be on disk.

### A.2 — How the policy was trained

**Within `rl-locomotion-learning` itself: nothing.** `grep`-ing the whole repo for training hyperparameters, reward terms, iteration counts, or domain-randomization ranges returns nothing. `README.md:29` states only: *"originally trained in Isaac Gym on a GPU cluster."* `stage2-go2-mujoco-inference/README.md` and `experiments/results/EXPERIMENT_FINDINGS.md:11` repeat "Isaac Gym" but add no parameters. There is no vendored training code, no config file, no training log in this repo.

**[external repo]** `robot-dog-sim/walk-these-ways-go2/runs/gait-conditioned-agility/pretrain-go2/train/142238.667503/parameters_cpu.pkl` contains the actual Isaac Gym `Cfg` object used for this run. Loading it (read-only, `torch.load(map_location='cpu')`) gives:

| Parameter | Value | Source |
|---|---|---|
| Algorithm | PPO (`clip_param=0.2, gamma=0.99, lam=0.95, entropy_coef=0.01, lr=1e-3`), wrapped in `RunnerArgs.algorithm_class_name = 'RMA'` | `PPO_Args`, `RunnerArgs` |
| Simulator | Isaac Gym, PhysX backend, GPU pipeline (`use_gpu_pipeline=True`), `sim.dt=0.005` (200 Hz), `physx.num_position_iterations=4`, `num_velocity_iterations=0` | `Cfg['sim']` |
| Control decimation | 4 (→ 0.005×4 = 0.02 s = 50 Hz policy rate — matches the 50 Hz claimed at deployment) | `Cfg['control']['decimation']` |
| **Control type** | **`'actuator_net'`** — torque is computed by a learned actuator network, **not** a textbook PD law | `Cfg['control']['control_type']`; confirmed against `go2_gym/envs/base/legged_robot.py:763-1054` |
| **Actuator net file** | `resources/actuator_nets/unitree_go1.pt` — a **Go1** motor model, used unmodified for the Go2 | `legged_robot.py:1046` |
| Nominal PD-like gains fed to that actuator net | `stiffness.joint=20.0`, `damping.joint=0.5` | `Cfg['control']` |
| Num. parallel envs | 6800 | `Cfg['env']['num_envs']` |
| Episode length | 20 s (`max_episode_length≈1001` steps) | `Cfg['env']` |
| Observation / action dim | 70 / 12 | `Cfg['env']` |
| Terrain | `terrain_proportions=[0,0,0,0,0,0,0,0,1.0]` → 100% flat, `curriculum=True` | `Cfg['terrain']` — this does corroborate the "flat-trained" claim used throughout Experiment 3 |
| Domain randomization | friction `[0.1, 3.0]`, restitution `[0.0, 0.4]`, added base mass `[-1.0, 3.0]` kg, motor strength `[0.9, 1.1]`, gravity `±1.0` (every 8 s), lag timesteps randomized (`lag_timesteps=6`); `push_robots=False`; `randomize_Kp_factor=False`, `randomize_Kd_factor=False` (gains were **not** randomized) | `Cfg['domain_rand']` |
| Reward terms actually weighted nonzero | `tracking_lin_vel=1.0`, `tracking_ang_vel=0.5`, `tracking_contacts_shaped_force=4.0`, `tracking_contacts_shaped_vel=4.0`, `jump=10.0`, `dof_pos_limits=-10.0`, `raibert_heuristic=-10.0`, `orientation_control=-5.0`, `collision=-5.0`, `action_smoothness_{1,2}=-0.1` each, `feet_slip=-0.04`, `feet_clearance_cmd_linear=-30.0`, plus small torque/dof-vel/dof-acc/lin-vel-z/ang-vel-xy penalties | `Cfg['reward_scales']` (full list has ~40 keys; most are 0) |
| Recorded `max_iterations` | **1500** | `RunnerArgs` |
| **Actual iterations reached** | **19,990** (timesteps ≈ 3.26 × 10⁹) | `outputs.log` tail, `runs/.../142238.667503/outputs.log` |

**Note the discrepancy in the last two rows** — the saved config's `max_iterations=1500` does not describe what actually happened; the run was clearly resumed/extended well past that. If you cite iteration count in the paper, cite 19,990, not the config default, and say so was reached across possibly multiple resumed runs (the config dump reflects only the launch-time value).

**Verdict: NOT DETERMINABLE from the audited repo.** Everything in the table above required reading a pickle in the external sibling repo. If a reviewer cannot access that repo, none of this is reproducible or even statable by them.

### A.3 — MuJoCo / Python / dependency versions

- **No `requirements.txt`, `pyproject.toml`, or `environment.yml` exists anywhere in the repo** (checked repo root and `stage2-go2-mujoco-inference/`). Dependency versions are not pinned anywhere in version control.
- `README.md:111-121` ("Tech Stack" table) gives only loose ranges: Python 3.10, MuJoCo "3.x", PyTorch "2.1+", Gymnasium "0.29+", TensorBoard "latest".
- What is actually installed in the local `.venv` right now (which is itself `.gitignore`d — see `.gitignore:1`):

  | Package | Installed version |
  |---|---|
  | Python | 3.10.12 |
  | mujoco | 3.8.0 |
  | torch | 2.11.0 |
  | numpy | 2.2.6 |
  | gymnasium | 1.2.3 |
  | glfw | 2.10.0 |

- **Verdict: NOT DETERMINABLE from the repo with any precision.** The table above is a snapshot of one developer machine's environment at audit time, not a committed artifact. A reviewer re-running this today with `pip install mujoco torch numpy` would get whatever the current latest releases are — which may not match the versions above, let alone whatever was actually used when `exp1`–`exp3` were originally run in May 2026 (there is no lockfile, so even the author cannot fully reconstruct that after the fact).

### A.4 — Go2 model file

- `stage2-go2-mujoco-inference/scenes/go2_model/go2.xml` — vendored (copied directly into the repo, not a symlink, not a git submodule).
- `README.md:29` states the source: *"Unitree Go2 MJCF (from `mujoco_menagerie`)"* — i.e., Google DeepMind's `mujoco_menagerie` project. `go2_flat.xml:2-4` has a matching comment: *"Self-contained: robot model lives in this repo... No symlinks, no external menagerie dependency. Fully reproducible."*
- **No commit hash, release tag, or date is recorded anywhere** for which version of `mujoco_menagerie`'s Go2 model was copied in. If DeepMind updates that model upstream (inertials, mesh, joint ranges), there is no way to tell which revision this repo's copy corresponds to.
- **Verdict: source identified (mujoco_menagerie), vendored (not fetched at runtime), but not version-pinned.**

### A.5 — Simulation parameters: explicit vs. MuJoCo default

Verified empirically by loading the actual compiled model (`mujoco.MjModel.from_xml_path`), not by reading the XML alone — this shows the *resolved* values MuJoCo runs with, defaults included.

| Parameter | Value | Explicit or default? | Where |
|---|---|---|---|
| Timestep | 0.002 s (500 Hz) | **MuJoCo default** — no `<option timestep>` anywhere in `go2.xml` or any scene file | resolved model, `m.opt.timestep` |
| Integrator | Euler (`0`) | **MuJoCo default** — not set | resolved model |
| Solver | Newton (`2`) | **MuJoCo default** — not set | resolved model |
| Solver iterations | 100 / `ls_iterations`=50 / tolerance=1e-8 | **MuJoCo defaults** — not set | resolved model |
| Contact cone | elliptic | **Explicitly set**, overriding MuJoCo's default pyramidal cone | `go2.xml:4` `<option cone="elliptic" impratio="100"/>` |
| `impratio` | 100 | **Explicitly set**, overriding default of 1 | `go2.xml:4` |
| Foot geom friction | `0.8 0.02 0.01` (sliding, torsional, rolling), `condim="6"` | **Explicitly set** | `go2.xml:31-34` (`class="foot"`) |
| Non-foot body geom friction | `0.6`, `condim="1"` (frictionless — normal force only) | **Explicitly set** (but functionally irrelevant since condim=1 discards friction) | `go2.xml:8` (`class="go2"` default) |
| **Ground / ramp / step geom friction** | `[1.0, 0.005, 0.0001]` | **MuJoCo built-in default** — the `floor`, `runup`, `ramp`, and `step_N` geoms in every scene file (`go2_flat.xml`, `go2_slope_*.xml`, `go2_stairs_*.xml`) declare **no `friction` attribute at all** | e.g. `go2_flat.xml:27`, `go2_stairs_05.xml:23-35` |
| Foot `solimp` | `0.015 1 0.022` (+ default width/power `0.5 2`) | **Explicitly set** | `go2.xml:32-33` |
| All `solref` (foot and floor) | `0.02 1` | **MuJoCo default everywhere** — no scene or model file sets `solref` | resolved model |
| Joint armature | 0.01 (all 12 joints) | **Explicitly set** | `go2.xml:9` |
| Joint (passive) damping | 2.0 (all 12 joints) — this is physical joint damping, distinct from the PD control-loop derivative gain | **Explicitly set** | `go2.xml:9` |
| Joint frictionloss | 0.2 (all 12 joints) | **Explicitly set** | `go2.xml:9` |
| Gravity | `[0, 0, -9.81]` | Matches Isaac Gym training config exactly (`Cfg['sim']['gravity']`) | resolved model |

**This directly answers the reviewer's "how much of the gap is parameter mismatch" question**: the floor and all terrain geometry are running at MuJoCo's *out-of-the-box default friction* (`μ=1.0`), never explicitly reasoned about or tuned — this is exactly the kind of "uncalibrated MuJoCo parameter" the paper's own limitations paragraph (Section IV.D) gestures at ("uncalibrated MuJoCo parameters that may account for part of the transfer gap") but doesn't specify. Now you can specify it.

### A.6 — Exact per-trial randomization ranges

All from `harness.py:149-166`, applied once at trial reset via a per-trial `np.random.default_rng(seed)`:

| Quantity | Range | Distribution | Line |
|---|---|---|---|
| Initial joint position offset (all 12 joints) | ±0.02 rad (≈ ±1.15°) | Uniform | `harness.py:154` |
| Initial base height offset | ±0.01 m | Uniform | `harness.py:156` |
| Initial base yaw | ±0.05 rad (≈ ±2.86°) | Uniform | `harness.py:158` |
| Initial joint velocity (all 12 joints) | ±0.05 rad/s | Uniform | `harness.py:166` |

No perturbation is applied to base x/y position, base pitch/roll, or base linear velocity. This matches Section III.E of the paper ("small randomized perturbations applied to the initial joint positions, base height, base yaw, and joint velocities") — the paper's qualitative description is accurate; it just doesn't give the numeric ranges, which the table above supplies.

### A.7 — Adaptation / fine-tuning / online updates

**Confirmed: strictly zero-shot, inference-only.** `grep -rniE "backward\(\)|optimizer|requires_grad|\.train\(\)|loss"` across every `.py` file in `stage2-go2-mujoco-inference/` (inference scripts and experiment harness alike) returns **zero matches**. Both networks are loaded via `torch.jit.load(...).eval()` and every forward pass runs inside `with torch.no_grad():` (`harness.py:210`). The adaptation module's output (`latent`) is a forward inference over the observation history, exactly as at deployment time in the original RMA/walk-these-ways design — it infers, it does not learn online. There is no gradient computation, no parameter write, anywhere in the deployment path.

---

## PART B — Numbers Ledger

Every number below is traced to the actual `exp{1,2,3}_*.csv` files (git-tracked, `stage2-go2-mujoco-inference/experiments/results/`), recomputed independently with the same aggregation the experiment scripts use (`numpy.mean()/.std()`, population `ddof=0`, over surviving trials only — matching `exp1_velocity_sweep.py:114-116` etc.), and cross-checked against the committed run logs (`results/run_logs/exp1_run_2026-05-18.txt`, `results/exp2_run_2026-05-18.txt`) and `EXPERIMENT_FINDINGS.md`. Those three sources (CSV / run log / findings doc) agree with each other everywhere I checked — they are one mutually consistent, reproducible chain. The comparison below is against the **paper's** printed tables and text.

**Headline finding before the detail: Table II (velocity) and Table III (gait) do not numerically match the committed CSVs, mostly in the standard-deviation column, in some cases by 3–10×. Tables IV and V (terrain) match closely.** See the bug list at the end.

### Table I (Deployment configuration) — all MATCH

| Claim | Paper value | Code source | Verdict |
|---|---|---|---|
| Degrees of freedom | 12 (3/leg) | 12 actuators in `go2.xml:188-201` | MATCH |
| Observation dimension | 70 | `harness.py:64` `OBS_DIM = 70`; concatenation in `build_obs` sums to 70 | MATCH |
| Observation history length | 30 | `harness.py:63` `HISTORY_LEN = 30` | MATCH |
| Physics rate | 500 Hz | resolved `m.opt.timestep = 0.002` | MATCH |
| Policy rate | 50 Hz | 500/10 | MATCH |
| Decimation | 10:1 | `harness.py:62` `DECIMATION = 10` | MATCH |
| Action type | Joint position residual | `harness.py:215` `joint_targets = DEFAULT_JOINT_POS + action * action_scale_per_joint` | MATCH |
| PD position gain | 25 | `harness.py:61` `KP = 25.0` | MATCH (to deployment code — **not** to training config, see bug list) |
| PD velocity gain | 0.6 | `harness.py:61` `KD = 0.6` | MATCH (to deployment code — **not** to training config, see bug list) |
| Trials per condition | 5 | `N_TRIALS = 5` in all three `exp*.py` | MATCH |

### Table II — Velocity transfer on flat ground

Recomputed directly from `exp1_velocity_sweep.csv` (30 rows, all survived):

| cmd (m/s) | Metric | Paper value | Recomputed from CSV (rounded to paper's precision) | Verdict |
|---|---|---|---|---|
| 0.00 | Achieved mean | 0.072 | 0.073 | **MISMATCH** |
| 0.00 | Achieved std | ±0.003 | ±0.001 | **MISMATCH** (3× off) |
| 0.00 | Tracking error | 0.072 | 0.073 | **MISMATCH** |
| 0.00 | Lateral drift | 0.043 | 0.043 | MATCH |
| 0.00 | Height std | 0.006 | 0.0065 (rounds either way) | MATCH (borderline) |
| 0.00 | Survival | 5/5 | 5/5 | MATCH |
| 0.25 | Achieved mean | 0.143 | 0.145 | **MISMATCH** |
| 0.25 | Achieved std | ±0.003 | ±0.003 | MATCH |
| 0.25 | Tracking error | 0.108 | 0.105 | **MISMATCH** |
| 0.25 | Lateral drift | 0.111 | 0.111 | MATCH |
| 0.25 | Height std | 0.005 | 0.006 | **MISMATCH** |
| 0.25 | Survival | 5/5 | 5/5 | MATCH |
| 0.50 | Achieved mean | 0.227 | 0.227 | MATCH |
| 0.50 | Achieved std | ±0.005 | ±0.005 | MATCH |
| 0.50 | Tracking error | 0.273 | 0.273 | MATCH |
| 0.50 | Lateral drift | 0.162 | 0.162 | MATCH |
| 0.50 | Height std | 0.006 | 0.006 | MATCH |
| 0.50 | Survival | 5/5 | 5/5 | MATCH |
| 0.75 | Achieved mean | 0.368 | 0.367 | **MISMATCH** |
| 0.75 | Achieved std | ±0.005 | ±0.004 | **MISMATCH** |
| 0.75 | Tracking error | 0.383 | 0.383 | MATCH |
| 0.75 | Lateral drift | 0.135 | 0.134 | **MISMATCH** |
| 0.75 | Height std | 0.007 | 0.006 | **MISMATCH** |
| 0.75 | Survival | 5/5 | 5/5 | MATCH |
| 1.00 | Achieved mean | 0.503 | 0.504 | **MISMATCH** |
| 1.00 | Achieved std | ±0.004 | ±0.001 | **MISMATCH** (4× off) |
| 1.00 | Tracking error | 0.497 | 0.496 | **MISMATCH** |
| 1.00 | Lateral drift | 0.018 | 0.018 | MATCH |
| 1.00 | Height std | 0.007 | 0.0065 (rounds either way) | MATCH (borderline) |
| 1.00 | Survival | 5/5 | 5/5 | MATCH |
| 1.50 | Achieved mean | 0.557 | 0.558 | **MISMATCH** |
| 1.50 | Achieved std | ±0.005 | ±0.010 | **MISMATCH** (2× off) |
| 1.50 | Tracking error | 0.943 | 0.942 | **MISMATCH** |
| 1.50 | Lateral drift | 0.348 | 0.347 | **MISMATCH** |
| 1.50 | Height std | 0.006 | 0.006 | MATCH |
| 1.50 | Survival | 5/5 | 5/5 | MATCH |

Only the **cmd = 0.50 m/s row is a clean, complete match.** Every other row has at least one mismatched cell, and the standard-deviation column is wrong (usually too large) almost everywhere except that one row.

### Table III — Gait performance at 0.50 m/s

Recomputed from `exp2_gait_robustness.csv` (15 rows, all survived):

| Gait | Metric | Paper value | Recomputed from CSV | Verdict |
|---|---|---|---|---|
| Trot | Tracking error | 0.273 ± 0.005 | 0.273 ± 0.005 | MATCH |
| Trot | Lateral drift | 0.162 ± 0.007 | 0.162 ± 0.006 | **MISMATCH** (std) |
| Trot | Height std | 0.006 ± 0.000 | 0.006 ± 0.000 | MATCH |
| Pace | Tracking error | 0.173 ± 0.004 | 0.173 ± 0.002 | **MISMATCH** (std, 2×) |
| Pace | Lateral drift | 0.086 ± 0.008 | 0.087 ± 0.006 | **MISMATCH** (mean & std) |
| Pace | Height std | 0.011 ± 0.001 | 0.011 ± **0.000** | **MISMATCH** — the raw CSV rows are `0.0109` for *all five* pace seeds; the true cross-seed std is exactly 0, not ±0.001 |
| Bound | Tracking error | 0.333 ± 0.007 | 0.333 ± 0.006 | **MISMATCH** (std) |
| Bound | Lateral drift | 0.202 ± 0.004 | 0.203 ± 0.002 | **MISMATCH** (mean & std) |
| Bound | Height std | 0.002 ± 0.000 | 0.002 ± 0.000 | MATCH |

Every mean in this table is right or within one rounding step. Every non-zero standard deviation in this table is wrong, and one row (pace height std) claims variance that provably does not exist in the underlying data — the CSV shows five bit-identical values.

### Table IV — Slope traversal — MATCHES (with citation for the 2.5 m run-up)

| Slope | Metric | Paper | CSV | Verdict |
|---|---|---|---|---|
| 5° | Survival, dist, vx | 5/5, 5.76±0.00, 0.284±0.001 | 5/5, 5.7626±0.0002, 0.2841±0.0013 | MATCH |
| 10° | Survival, dist, vx | 5/5, 5.43±0.01, 0.269±0.000 | 5/5, 5.4250±0.0099, 0.2695±0.0002 | MATCH |
| 15° | Survival, dist, vx | 1/5, 4.75 (1 trial), 0.221 (1 trial) | 1/5, 4.7544, 0.2211 | MATCH |
| 20°, 25° | Survival | 0/5, "–" | 0/5, all fell | MATCH |

The paper's "run-up length 2.5 m" (Section III.E, used to define the 3.5 m progress threshold) is exactly `2 × 1.25` — the half-length of the `runup` box geom declared in every terrain scene, e.g. `go2_stairs_05.xml:24`: `size="1.2500 2.0 0.05"`. MATCH, and now cited.

### Table V — Stair traversal — MATCHES

| Step | Metric | Paper | CSV | Verdict |
|---|---|---|---|---|
| 2 cm | dist, vx | 4.02±0.09, 0.034±0.013 | 4.0165±0.0856, 0.0335±0.0125 | MATCH |
| 5 cm | dist, vx | 0.47±0.01, 0.024±0.000 | 0.4701±0.0075, 0.0235±0.0005 | MATCH |
| 8 cm | dist, vx | 0.47±0.01, 0.024±0.000 | 0.4704±0.0068, 0.0235±0.0005 | MATCH |
| 12 cm | dist, vx | 0.47±0.01, 0.024±0.000 | 0.4702±0.0072, 0.0235±0.0005 | MATCH |
| 16 cm | dist, vx | 0.48±0.00, 0.024±0.000 | 0.4759±0.0046, 0.0237±0.0004 | MATCH |

Note for Part C: seeds 1–4 produce **bit-identical** `mean_vx`/`distance_traveled` across the 5/8/12 cm conditions. This is a real physical effect, not a broken seed — see Part C.1.

### Abstract / Conclusion / narrative numeric claims

| # | Claim | Location | Traced to | Verdict |
|---|---|---|---|---|
| 1 | "saturates near 0.55 m/s" | Abstract | max achieved vx in CSV = 0.5577 (cmd=1.5); Table II shows 0.503–0.558 plateau | MATCH (qualitative "near", reasonable) |
| 2 | "forward velocity saturates near 0.55 m/s" | Conclusion (repeat of #1) | same | MATCH |
| 3 | "posture remains stable across all tested speeds" | Abstract | height std 0.0060–0.0065 across all 6 cmd values (CSV) | MATCH |
| 4 | "body-height std... 0.005 to 0.007 m" range | Results IV.A | actual per-condition CSV values round to 0.006 in 5 of 6 rows, 0.006–0.007 in the last — the *paper's own table* shows 0.005 at cmd=0.25 which the CSV does not reproduce (see Table II mismatch above) | **MISMATCH** — same underlying issue as Table II |
| 5 | "all twenty-five non-zero trials remained stable" | Results IV.A | 5 non-zero velocities × 5 seeds = 25, all `fell=False` in CSV | MATCH |
| 6 | "all fifteen trials remained stable" | Results IV.B | 3 gaits × 5 seeds = 15, all `fell=False` in CSV | MATCH |
| 7 | "pace attains lower velocity-tracking error than trot at moderate speed" | Abstract | 0.173 (pace) < 0.273 (trot), CSV confirms | MATCH |
| 8 | "traverses inclines up to 10 degrees" | Abstract/Conclusion | Table IV, confirmed | MATCH |
| 9 | "steps up to 2 cm" | Abstract/Conclusion | Table V, confirmed | MATCH |
| 10 | Progress threshold "3.5 m" | Section III.E, IV.C | `exp3_terrain.py:91` `PROGRESS_THRESHOLD_M = 3.5` | MATCH |
| 11 | Run-up length "2.5 m" | Section IV.C | `2 × 1.25` from scene geometry, see Table IV note above | MATCH |
| 12 | "1 m of genuine terrain progress" (3.5 = 2.5 + 1) | Section IV.C | arithmetic, consistent with #10/#11 | MATCH |
| 13 | Time-to-fall "approximately 18 s at 20 degrees" | Results IV.C.1 | CSV `go2_slope_20.xml` fall times: 19.05, 17.75, 18.07, 18.21, 17.75 → mean 18.17 s | MATCH |
| 14 | Time-to-fall "15 s at 25 degrees" | Results IV.C.1 | CSV `go2_slope_25.xml` fall times: 16.15, 15.19, 15.13, 15.15, 15.14 → mean 15.35 s | MATCH |
| 15 | "at least four of five trials not falling" (traversability survival criterion) | Section III.E | `exp3_terrain.py:97` `surv_rate >= 0.8` | MATCH |
| 16 | "All gains, scaling factors, history length, and stepping frequency match the training configuration" | Section III.C | **False as stated for the gains** — see bug list below | **MISMATCH / unsupported claim** |

---

## PART C — Harness and Seeds

### C.1 — Do the five trials per condition genuinely differ? What does a reported std of 0.000 mean?

**Yes, seeds genuinely reach the randomization and genuinely produce different trajectories** — confirmed directly in the raw CSV, e.g. `exp1_velocity_sweep.csv` rows for `cmd_vx=0.0`: `mean_vx` = 0.0709, 0.0738, 0.0736, 0.0733, 0.0736 across seeds 0–4 — five distinct floating-point values, not five copies of one number.

The apparent "0.000" standard deviations are real but come in two different flavors, and it matters which:

1. **Genuine near-zero variance in a stable quantity.** `height_std` (the within-trial std of body height over the 30 s measurement window) is reported as *identical to 4 decimal places across all 5 seeds* for every velocity in `exp1_velocity_sweep.csv` (e.g. all five `cmd_vx=0.0` rows show `height_std=0.0065` exactly). This is physically sensible: body-height oscillation during a stable trot is a limit-cycle property of the gait, and the ±0.02 rad / ±0.01 m initial perturbations decay out during the 3 s settle window (`harness.py:190-192`, `SETTLE_S=3.0`) well before measurement starts. This is genuine determinism of a converged quantity, not a broken seed.
2. **Genuine saturation from converged/stalled dynamics.** In `exp3_terrain.csv`, seeds 1–4 give bit-identical `mean_vx` and near-identical `distance_traveled` across the 5 cm, 8 cm, and 12 cm stair conditions (e.g. `mean_vx=0.0238` for seed 1 in all three). This is because the robot stalls against the first step almost immediately in all three cases — the step height doesn't matter once the robot can't climb any of them, so the seed-dependent initial-condition noise has already been washed out by the time it matters. Again genuine, not a bug — but it does mean the "5 cm / 8 cm / 12 cm" conditions in Table V are not really 3 independently informative data points; they're closer to 1 physical outcome (stall) sampled 3 times under different step geometry that happens not to matter yet.

Neither case indicates the seed is "never consulted" — `rng = np.random.default_rng(seed)` at `harness.py:152` is called fresh every trial and its output measurably perturbs `data.qpos`/`data.qvel` before `mujoco.mj_forward` is called. The zero variance is a property of the physics, not the code.

### C.2 — How is the seed set and threaded through?

**Per-trial, local, not global.** `harness.py:152`: `rng = np.random.default_rng(seed)`, a new NumPy `Generator` instance created inside `run_trial()` on every call, seeded from the `seed` argument passed in by the calling `exp*.py` script (`for seed in range(N_TRIALS): ...`). There is **no** call to `np.random.seed()` anywhere in the codebase — no global RNG state is touched. This is good practice: trial order and trial count don't affect any individual trial's outcome, and trials are safely parallelizable (though nothing in the repo currently does so — see Part C.4/D).

### C.3 — What does a trial write to disk?

**Summary-only. No per-timestep log is ever written or kept.** Inside `run_trial()` (`harness.py:194-236`), `vx_log`, `vy_log`, `h_log` are plain Python lists accumulated in memory only during the measurement window; they are reduced to scalars (`mean_vx`, `height_std`, etc.) and the lists themselves are discarded when the function returns (`harness.py:238-271`). The full `MjData` trajectory, per-step contact states, and per-step joint torques are never captured anywhere. What actually reaches disk is one CSV row per trial, written by the calling script (e.g. `exp1_velocity_sweep.py:83-90`).

Raw results **are** committed to git — confirmed via `git ls-files`:
- `stage2-go2-mujoco-inference/experiments/results/exp{1,2,3}_*.csv`
- `stage2-go2-mujoco-inference/experiments/results/exp2_run_2026-05-18.txt`
- `stage2-go2-mujoco-inference/experiments/results/run_logs/exp1_run_2026-05-18.txt` (no equivalent log for exp3 is committed)
- `EXPERIMENT_FINDINGS.md`

### C.4 — Wall-clock cost and re-run feasibility at 30 seeds

**Not measured anywhere in the repo, and not reasonably re-measurable without running code (which I was told not to do).** `grep`-ing `harness.py` and all three `exp*.py` scripts for `time.time`, `perf_counter`, or any timing instrumentation returns nothing — there is no logged per-trial or per-experiment duration anywhere in the repo, and no timestamped log entries fine-grained enough to back one out reliably (the file-modification-time gaps between `exp1`/`exp2`/`exp3` CSVs span from ~4 minutes to ~3.3 hours, but those gaps include debugging and manual iteration — recall `EXPERIMENT_FINDINGS.md` documents the metric being corrected mid-Experiment-3 — not pure compute time, so they cannot be trusted as a clean per-trial benchmark).

What I *can* state precisely, from the code:
- **The harness is single-process and single-threaded.** No `multiprocessing`, `joblib`, `concurrent.futures`, or `torch.set_num_threads` call exists anywhere in `experiments/`. Every trial runs strictly sequentially in one Python process, even though trials are fully independent (per C.2) and this is an embarrassingly parallel workload.
- **This machine has 16 logical cores** (`nproc` = 16, Intel Core i5-13400F) sitting unused by the current scripts.
- Exact step counts per trial (deterministic, from `dt=0.002` and each script's `settle_s`+`measure_s`):
  - Exp 1 & 2: 33 s / 0.002 s = 16,500 steps/trial, 1,650 policy-network forward passes/trial (decimation 10).
  - Exp 3: 23 s / 0.002 s = 11,500 steps/trial nominal (shorter for any trial that falls and breaks early).

Given the missing timing data, I will not fabricate a specific "X seconds per trial" number for the paper — that would be exactly the kind of guess you told me not to make. **Before you write anything about compute cost in the rebuttal, run one real trial and time it**, e.g.:
```
cd stage2-go2-mujoco-inference/experiments && time python3 harness.py
```
(this only runs the built-in single-trial smoke test at the bottom of `harness.py:274-282` — it prints one JSON dict and writes nothing to disk).

Once you have that single-trial number `T`, the estimate for a 30-seed re-run of everything is mechanical: current 5-seed total is 95 trials (30+15+50); at 30 seeds it becomes 570 trials (180+90+300), i.e. **6× the current trial count**, run serially. If a single trial takes on the order of a few seconds to a couple of minutes (typical for a 12-DOF MuJoCo model with two small `torch.jit` nets on a modern desktop CPU with no rendering), 570 sequential trials plausibly lands somewhere between roughly 20 minutes and several hours — the 3-hour bound genuinely could go either way depending on `T`, which is exactly why I'm not guessing it. If `T` turns out to be more than ~19 seconds, serial execution alone will blow past 3 hours; parallelizing across the 16 idle cores (trivial to add — trials are independent, seeds don't interact) would divide that by close to 16×, almost certainly bringing any realistic `T` back under 3 hours. **The actual bottleneck for your deadline is not compute, it's that the harness doesn't parallelize at all yet — that's a ~30-minute code change (Part D.1), not a compute problem.**

---

## PART D — Feasibility for the Rebuttal

### D.1 — 30 trials/condition, mean/SD/95% CI, pairwise significance between gaits

- **What exists:** `run_trial()` is already deterministic-per-seed and trivially callable with `seed=5..29`. `exp1_velocity_sweep.py`/`exp2_gait_robustness.py`/`exp3_terrain.py` already loop `for seed in range(N_TRIALS)` — raising `N_TRIALS` to 30 is a one-line change in each of the three files.
- **What's missing:** No CI computation and no significance testing exists anywhere in the repo (no `scipy.stats` import, no `t.ppf`, no hypothesis test of any kind). Also no parallelism (Part C.4) — at 30 seeds you're running 6× the trials serially unless you add it.
- **Effort:**
  - Bump `N_TRIALS` in 3 files + add a `multiprocessing.Pool` (or just run the 3 scripts concurrently in separate processes, trivial since they don't share state) — **~30–45 min**.
  - Add mean/SD/95% CI (t-distribution, since n≤30) and pairwise Welch's t-test (unequal variances are likely given e.g. bound's near-zero height-std vs pace's) between trot/pace/bound per metric — **~1–1.5 hrs** of straightforward `scipy.stats.ttest_ind(..., equal_var=False)` code plus a results table.
  - Actual compute time: unknown per C.4, but parallelized across 16 cores this should not be the bottleneck.
  - **Total: roughly half a day of engineering time**, dominated by writing and sanity-checking the stats code, not by simulation time.

### D.2 — Friction/contact-stiffness sensitivity sweep at fixed command

- **What exists:** `run_trial(scene_path, ...)` already takes a `scene_path`; MuJoCo's Python API lets you mutate `model.geom_friction[...]` and `model.geom_solref[...]`/`model.geom_solimp[...]` in-memory after loading, without touching any XML file — no new scene files needed.
- **What's missing entirely:** No sweep code exists. Nothing in the repo currently varies any contact parameter. You'd write a new script (pattern-matched off `exp1_velocity_sweep.py`) that: loads `go2_flat.xml` once, loops over a friction grid (e.g. `[0.3, 0.6, 1.0, 1.5, 2.0]` — bracketing both the robot's own 0.6/0.8 and the ground's default 1.0 found in A.5) and a `solref`/contact-stiffness grid, mutates `model.geom_friction`/`model.geom_solref` for the relevant geoms before each trial batch, and records the same metrics `run_trial` already returns.
- **Effort:** Writing the sweep harness (parameterizing model mutation, reusing `run_trial`) — **~2–3 hrs**. This is genuinely new code, not a config change, but it's a straightforward generalization of what's already there — you are not writing a new simulator interface, just a loop that pokes two arrays before calling the existing function.
- This is the single most valuable thing you could add for the specific reviewer comment about parameter-mismatch vs. genuine simulator difference, precisely because A.5 already shows the ground friction was never touched — a friction sweep would let you say something quantitative about that rather than speculating.

### D.3 — Per-timestep log for one representative trial per gait (foot-contact / gait-phase plot)

- **What exists:** `build_obs()` already computes the 4-dimensional gait-phase clock signal every policy step (`harness.py:106-112`) — the phase values are already sitting in a local variable, just never persisted.
- **What's missing:** `run_trial()` discards all per-step data (Part C.3). It needs a new optional path (e.g. `log_path=None` kwarg) that, when set, appends per-step `(t, qpos, qvel, clock, contact_geom_ids)` to an array/CSV instead of only accumulating the three summary lists.
- Foot contact extraction is not currently done anywhere (no code reads `data.contact` in the whole `experiments/` directory) — you'd need to iterate `data.contact[:data.ncon]` each step and check `geom1`/`geom2` against the 4 foot geom IDs (`FL`, `FR`, `RL`, `RR` — named exactly in `go2.xml:100,127,154,181`) to get a boolean stance/swing signal per foot.
- **Effort:** Minimally-invasive `harness.py` change to add opt-in logging — **~1–1.5 hrs**. Foot-contact boolean extraction — **~1 hr**. A plotting script (phase + contact state vs. time, one panel per gait) — **~1 hr**. **Total: half a day.** Only needs to be run once per gait (3 trials, ~33 s of sim each) — trivial compute cost regardless of the C.4 unknowns.

---

## Contradictions and Bugs — Blunt Summary

1. **The paper's claim that deployment gains "match the training configuration" (Section III.C) is contradicted by the actual training config.** Deployment uses `KP=25, KD=0.6` (`harness.py:61`, Table I). The actual Isaac Gym `Cfg['control']` for this checkpoint records `stiffness.joint=20.0, damping.joint=0.5` (Part A.2). These are not the same numbers. Worse, training didn't use a plain PD torque law at all — `control_type='actuator_net'` routes the position/velocity error through a **Go1** actuator network (`unitree_go1.pt`) before it becomes torque; the 20/0.5 values are just the *target* fed into that network, not the effective gain. So the honest statement is: deployment approximates an unknown, Go1-actuator-network-mediated torque response with an idealized PD controller using gains that don't match even the nominal training targets. This is directly relevant to the reviewer's parameter-mismatch question and currently misstated in the manuscript.

2. **Tables II and III do not reproduce from the repository's own committed data.** The CSVs, run logs, and `EXPERIMENT_FINDINGS.md` all agree with each other (this pipeline is internally reproducible) but disagree with the paper in most cells, overwhelmingly concentrated in the reported standard deviations (off by 2–10×, and in one case — pace height-std in Table III — claiming variance that is provably zero in the data: all five seeds give the identical value `0.0109`). Table II's mean values also drift by ±0.001–0.002 in 5 of 6 rows. Only the cmd=0.5 m/s row of Table II is a full, exact match. By contrast, Tables IV and V (terrain) match the CSV closely in every cell. Since seeded MuJoCo + frozen-net inference is fully deterministic given identical code and initial state (confirmed by internal consistency of CSV/run-log/findings.md), this cannot be attributed to trial-to-trial nondeterminism — it looks like Tables II/III were populated from a different, unlogged run of `exp1`/`exp2`, or hand-estimated rather than pulled mechanically from the script's own summary output. **This needs to be resolved before resubmission** — either regenerate Tables II/III from the current committed CSVs (the numbers you already have), or locate and commit whatever run actually produced the printed table values.

3. **The pretrained checkpoint is not reproducible by anyone but the author.** It's referenced by a hardcoded, machine-specific absolute path (`harness.py:19-22` and duplicated in 4 other files), it is git-ignored in both the repo that trained it and (trivially) absent from the repo being audited, and there is no download step, release artifact, or documented recovery path. If a reviewer or future collaborator tries to run this code today, it fails immediately with a missing-file error and no indication of where to get the file.

4. **No dependency lockfile exists anywhere** (Part A.3) — not even a loose `requirements.txt`. "MuJoCo 3.x, PyTorch 2.1+" in the README is not a pin; it's a hope. You cannot currently guarantee that re-running this in six months reproduces the same floating-point behavior, let alone that a reviewer's environment does.

5. **The Go2 model provenance is unpinned** (Part A.4) — sourced from `mujoco_menagerie` per the README, vendored directly, but with no commit hash or version recorded for which snapshot was copied.

6. **`MUJOCO_LOG.TXT`** (present locally, `.gitignore`d) shows multiple `NaN/Inf in CTRL` instability warnings on several dates, including one dated after the final Experiment 3 commit. None of the three experiment CSVs contain NaN values, so this doesn't appear to have corrupted the reported results — but it does confirm the simulation is capable of numerically blowing up under some conditions (most likely during interactive/manual runs via `07_run_policy_interactive.py`, not the batch harness), which is worth being aware of if you extend the velocity sweep beyond 1.5 m/s for the rebuttal.

7. **Minor:** the paper's Section III.D says "the decimation between the physics and policy rates must match the training value" — literally false (training decimation is 4 at `dt=0.005`; deployment decimation is 10 at `dt=0.002`). What actually matches is the *resulting* 50 Hz policy control frequency, not the decimation integer itself. Harmless as a design choice, but the sentence as written overclaims precision it doesn't have.
