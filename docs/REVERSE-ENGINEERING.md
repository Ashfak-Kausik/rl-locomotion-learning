# Reverse Engineering Report

What is *actually* in this codebase, derived by reading every file rather than
trusting the READMEs. Written for a newcomer who needs to know where the solid
ground is before they start building.

**Method:** every `.py`, `.xml`, `.csv` and `.md` file read in full; all
constants cross-checked against `scenes/go2_model/go2.xml`; every claim in the
READMEs checked against the code that would have to produce it; the full
pipeline executed on a clean machine (host venv and Docker).

**Verdict up front:** the research is sound and the claims check out. The
engineering had one hard blocker (machine-specific absolute paths, now fixed)
and carries the normal debris of a codebase that grew as a learning exercise —
superseded files kept for history, constants duplicated across scripts, a few
stale comments. Nothing here is alarming; all of it is worth knowing.

---

## Contents

1. [Inventory](#1-inventory)
2. [Findings — blockers](#2-findings--blockers-fixed-in-this-branch)
3. [Findings — correctness traps](#3-findings--correctness-traps)
4. [Findings — hygiene](#4-findings--hygiene)
5. [The duplication question](#5-the-duplication-question)
6. [Claim verification](#6-claim-verification)
7. [What is implemented](#7-what-is-implemented)
8. [What is not implemented](#8-what-is-not-implemented)
9. [Suggested first contributions](#9-suggested-first-contributions)

---

## 1. Inventory

| Area | Files | Lines | Status |
|---|---|---|---|
| Theory chapters | 8 md | ~1,090 | complete, self-contained |
| Stage 1 (RL fundamentals) | 6 py | 169 | complete, all 3 envs solved |
| Stage 2 (inference) | 8 py | 1,332 | complete, robot walks |
| Experiments | 7 py | 1,111 | complete, 3 studies, 75 trials |
| Scenes | 11 xml + 16 obj | — | self-contained, regenerable |
| Results | 3 csv + findings md | 75 rows | committed |
| Figures | 4 data + 7 qualitative | — | committed |

~2,600 lines of Python. Small, readable, no framework — deliberately so.

---

## 2. Findings — blockers (fixed in this branch)

### R1 — Hardcoded absolute paths made the repo unrunnable off one machine

**Severity: critical.** Eight of nine Stage 2 scripts opened paths under
`/home/user/projects/...`:

```python
MODEL_PATH = "/home/user/projects/mujoco_menagerie/unitree_go2/scene.xml"
POLICY_DIR = "/home/user/projects/robot-dog-sim/walk-these-ways-go2/runs/..."
```

Neither path exists on any other computer. A fresh clone could not run a single
Stage 2 script, and containerisation was impossible.

**Fix:** added [`stage2-go2-mujoco-inference/paths.py`](../stage2-go2-mujoco-inference/paths.py),
which resolves both from environment variables with in-repo defaults, and
replaced the ten constant definitions with imports from it.

```python
GO2_SCENE       # scene filename inside scenes/   (default go2_flat.xml)
GO2_MODEL_PATH  # full path to a scene XML        (overrides GO2_SCENE)
GO2_POLICY_DIR  # dir holding the two .jit files
```

No physics or control logic changed — only which files get opened.

**Behavioural note worth knowing:** the default model path changed from the
external `mujoco_menagerie/unitree_go2/scene.xml` to the in-repo
`scenes/go2_flat.xml`. Both wrap the same `go2.xml` robot on a flat ground
plane, so the simulation is equivalent, and the in-repo version is the one
every experiment already used.

### R2 — Cryptic failure when the policy weights are absent

Missing checkpoints produced a bare `ValueError: The provided filename ... does
not exist`, with the old path pointing at a stranger's home directory.

**Fix:** `paths.py` exposes `require_policy()`, which raises a message naming
the files, the search location, the env var to set, and — importantly — the
list of things that still work without them. `harness.py` calls it, so all
three experiments now fail informatively.

### R3 — Stray import that breaks headless runs

`stage1-rl-fundamentals/02_cartpole_watch.py` opened with:

```python
from turtle import done      # unused; pulls in tkinter
```

An accidental editor auto-import. Harmless on a desktop, an `ImportError` in
any container without tkinter. **Fix:** removed.

---

## 3. Findings — correctness traps

These are **not bugs**, but each one will mislead a newcomer.

### R4 — `04_build_obs_vector.py` contains a superseded, incorrect layout (fixed)

The file is a preserved learning artefact from *before* the observation layout
was recovered correctly, and it disagrees with the working code in two ways:

| | `04_build_obs_vector.py` (wrong) | `harness.py` / `06` / `07` / `08` (correct) |
|---|---|---|
| Field order | grav, cmd_vel(3), joints, vels, action, **clock**, gait_cmd(8), body_cmd(3), tail(13) | grav, **cmd(15)**, joints, vels, action, prev_action, **clock last** |
| Clock position | indices 42–46 | indices **66–70** |
| Hip signs | FL `−0.1`, FR `+0.1` | FL `+0.1`, FR `−0.1` |
| Foot phases | `[p, p+0.5, p+0.5, p+0.5]` | `[t+ph+off+bd, t+off, t+bd, t+ph]` |

Both errors are exactly the ones the Stage 2 README lists as "key debugging
insights", so the file is genuinely valuable *as history*. It is dangerous only
if you copy from it.

**Fix applied:** header banner added marking the file superseded, pointing to
`harness.py` as the reference implementation. Also fixed the `[42:54]`
comment (R8) to `[42:46]`, matching the file's own (wrong) concatenation —
the file's content is deliberately left otherwise unchanged, it's the
history that's valuable.

### R5 — A misleading comment about hip indices (fixed)

`06_run_policy.py`:

```python
# Hip-scale-reduction mask (FR_hip, FL_hip, RR_hip, RL_hip are indices 0,3,6,9 ...)
```

The **indices were right**; the **names were in the wrong order**. Verified
against `scenes/go2_model/go2.xml` lines 189–200, the actuator order is
`FL, FR, RL, RR`, so indices 0/3/6/9 are `FL_hip, FR_hip, RL_hip, RR_hip`.
Comment-only defect — the mask itself was always correct. **Fixed** by
correcting the leg names in the comment to match.

### R6 — MuJoCo silently needs absolute scene paths

Passing a *relative* path to `mujoco.MjModel.from_xml_path()` breaks mesh
resolution for these scenes:

```
ValueError: Error: Error opening file 'stage2-.../scenes/go2_model/base_0.obj'
```

...even though the file exists. An absolute path works. Every script in the
repo already wraps its path in `os.path.abspath()`, which is presumably why
this was never hit — but it is invisible convention, not enforced. `paths.py`
now always produces absolute paths.

### R7 — Two different "default poses" exist and both are correct

The MJCF `home` keyframe (`0, 0.9, −1.8` per leg) is not the training default
pose (`±0.1, 0.8/1.0, −1.5`). Every working script resets to the keyframe and
then overwrites `qpos[7:19]`. Skipping the override silently offsets all 12
joint observations. This is documented in the READMEs and correctly implemented
everywhere except the superseded `04`.

### R14 — The screenshot feature never worked (fixed)

**Found by actually running the project**, not by reading it.

`08_view_scene.py` built `mujoco.Renderer(model, height=1080, width=1920)`,
but no scene declares `offwidth`/`offheight`, so MuJoCo's default 640x480
offscreen framebuffer applied. `Renderer` **refuses** a larger request rather
than resizing it, so every screenshot raised:

```
ValueError: Image width 1920 > framebuffer width 640.
```

The script exists specifically to capture paper figures, so its one job was
broken from the day it was written. It went unnoticed because triggering it
requires the policy checkpoints, which are missing on any machine but the
author's.

**Fix:** `SHOT_WIDTH, SHOT_HEIGHT` constants, and the model's framebuffer is
enlarged in `main()` before any `Renderer` is constructed. Verified by
rendering a real 1920x1080 frame. Pinned by
`test_screenshot_resolution_fits_the_framebuffer`.

**Worth generalising:** a test suite that only exercises importable code would
never have caught this. Running the thing found it in minutes.

### R8 — Stale comments (fixed)

- `06_run_policy.py:243` — `# run for 30 seconds` above `< 60`. Fixed to say
  60.
- `04_build_obs_vector.py` — index comment said `[42:54] clock` but the
  concatenation beneath it places a 4-element clock at 42, i.e. `[42:46]`.
  Fixed to match (still labeled wrong-vs-harness via the R4 banner — only the
  arithmetic was corrected).

Cosmetic, but they are exactly what a newcomer trusts.

---

## 4. Findings — hygiene

### R9 — No dependency manifest (fixed)

There was no `requirements.txt`, no lockfile, no environment file. Versions had
to be inferred from imports. **Fixed:** `requirements.txt`, `scripts/setup_env.sh`,
`scripts/check_env.py`, and a Docker image.

### R10 — `.gitignore` had duplicated entries

`tb_logs/` and `*.zip` each appeared three times. Rewritten and commented, with
a `policies/**` + `!policies/README.md` rule added for the checkpoints.

### R11 — `.vscode/` was committed despite being ignored

`.vscode/browse.vc.db*` (IntelliSense caches) are tracked in git history from
before the ignore rule was added. Not fixed here — untracking them is a
history-touching decision for the repo owner.

### R12 — Root README is truncated

`README.md` ends at the heading `## Repository Structure` with no content
beneath it. The stage roadmap above it is complete and accurate.

### R13 — Absolute paths in committed run logs

`experiments/results/*_run_*.txt` contain the original machine's paths. These
are *historical records* of actual runs and should be left exactly as they are.

---

## 5. The duplication question

`DEFAULT_JOINT_POS`, `OBS_SCALES`, `COMMANDS_SCALE`, `quat_rotate_inverse`,
`build_obs`, `GAIT_PRESETS` and the PD constants are copy-pasted across
`06_run_policy.py`, `07_run_policy_interactive.py`, `08_view_scene.py` and
`experiments/harness.py`. A linter would flag this immediately.

**Do not "fix" it reflexively.** The numbered scripts are a *curriculum*: each
is meant to be read top-to-bottom as a complete, standalone artefact. Extracting
a shared module would improve the engineering and damage the teaching.

The pragmatic reading:

| File | Treat as | Refactor? |
|---|---|---|
| `01`–`05` | teaching artefacts | **no** — freeze them |
| `06` | the canonical walking demo | no |
| `07`, `08` | interactive tools | maybe — they are tools, not lessons |
| `harness.py` | production research code | **yes** — this is the reference |

If you consolidate, do it as `stage2-go2-mujoco-inference/go2_common.py`
imported by `07`, `08` and `harness.py`, leaving `01`–`06` untouched. The four
copies of `build_obs` are currently **semantically identical** (verified
line-by-line) apart from `06` reading a module-level `COMMANDS` constant while
the others take a parameter — so the risk today is drift, not disagreement.

---

## 6. Claim verification

| Claim (from README / findings) | Verified how | Result |
|---|---|---|
| "70-dim observation vector" | summed the concatenation: 3+15+12+12+12+12+4 | ✅ 70 |
| "2100-dim history" | `HISTORY_LEN 30 × OBS_DIM 70` | ✅ 2100 |
| "2102-dim body input" | `2100 + 2` latent | ✅ 2102 |
| "19 qpos, 18 qvel, 12 actuators" | loaded the model, printed `nq/nv/nu` | ✅ 19/18/12 |
| "Policy 50 Hz, sim 500 Hz, decimation 10" | `timestep 0.002` × `DECIMATION 10` | ✅ 50 Hz |
| "Hip signs FL/RL +, FR/RR −" | `go2.xml` actuator order vs `DEFAULT_JOINT_POS` | ✅ consistent |
| "Action scale 0.25, 0.5× for hips" | `action_scale_per_joint` at indices 0,3,6,9 | ✅ correct |
| "Self-contained Go2 model, no menagerie" | 16 `.obj` referenced, 16 present | ✅ complete |
| "Terrain scenes are reproducible" | re-ran the generator, `git diff` | ✅ byte-identical |
| "Figures regenerate from CSVs" | ran `make_figures.py` with no policy | ✅ all 4 |
| "~0.28 m/s at commanded 0.5" (README) | Exp 1 CSV reports 0.227 ± 0.005 | ⚠️ see below |
| "Robot walks 60s+ without failure" | Exp 1: 100% survival, 30 s windows | ✅ consistent |
| "Inference < 2 ms/step on CPU" | not re-measured | ⏸ plausible, unverified |

**The one discrepancy:** the root README's headline figure of **~0.28 m/s** does
not match Experiment 1's measured **0.227 ± 0.005 m/s** at the same commanded
0.5 m/s. The likely explanation is that the README number predates the
experiment harness — it came from the interactive `06_run_policy.py` run, which
has no settle window, no seeded initial-condition randomisation, and no
averaging over 5 trials. The harness number is the rigorous one.

**Resolved.** The README now reports 0.227 ± 0.005 m/s, cites
`exp1_velocity_sweep.csv`, states the measurement protocol (5 seeds, 3 s settle
+ 30 s window), and carries a note explaining why the earlier figure differed.
The qualitative claim is unchanged and is in fact the project's headline
finding. The Stage 1 result tables were reconciled the same way — their
timestep counts had disagreed with the scripts (25k vs 100k, 300k vs 1M,
400k vs 300k).

---

## 7. What is implemented

### Stage 1 — RL fundamentals ✅

- PPO on CartPole-v1 (discrete), LunarLander-v3 (shaped reward, 8 vec envs),
  Pendulum-v1 (continuous, gSDE)
- Train/save/load/evaluate/render cycle for each
- TensorBoard logging with multi-run comparison
- Documented analysis of policy collapse and convergence diagnostics

### Stage 2 — Go2 inference ✅

- MJCF loading, model introspection, PD pose holding
- 70-dim observation reconstruction with all scale factors and sign conventions
- 30-frame history buffer feeding an RMA adaptation module
- TorchScript inference: adaptation → concat → body → 12 joint deltas
- Dual-rate control (50 Hz policy / 500 Hz PD torque)
- Gait conditioning: trot, pace, bound
- Interactive keyboard teleoperation (`07`)
- Scene-selectable viewer with offscreen screenshot capture (`08`)

### Research layer ✅

- Headless trial harness with seeded initial-condition randomisation
- Fall detection (height < 0.15 m or tilt > 60°)
- Parametric terrain generation: 5 slopes, 5 stair heights
- Experiment 1 — velocity sweep, 6 × 5 trials
- Experiment 2 — gait robustness, 3 × 5 trials
- Experiment 3 — terrain robustness, 10 × 5 trials
- **A corrected traversability metric** — survival alone misclassified a robot
  safely stalling at the base of a step as a success; the metric now requires
  survival **and** > 3.5 m progress. This correction is a genuine
  methodological contribution and is documented as such.
- Figure pipeline: 4 data figures + 7 curated qualitative panels

### Infrastructure ✅ (added in this branch)

- `requirements.txt`, `scripts/setup_env.sh`, `scripts/check_env.py`
- Docker image + 4 compose services (lab / headless / viewer / tensorboard)
- `paths.py` portability layer
- This documentation set

---

## 8. What is not implemented

Ordered by how soon it matters.

### Stage 3 — custom policy training ⚙️ RESOLVED

This was the largest gap between the roadmap and the tree. It is now
implemented in `stage3-go2-training/`: a contract-compliant environment, 12
reward terms, a 7-level curriculum derived from Experiment 3's measured
failure boundaries, 8-parameter domain randomisation, PPO plus RMA phase-2
distillation, checkpoint/resume, TorchScript export and evaluation against the
baseline through Stage 2's own harness.

What remains is **compute, not code**: ~600 policy steps/s on 8 CPU cores,
against the 10⁸–10⁹ environment steps legged locomotion typically needs. The
MJX/GPU port is designed and contained to one file. See the
[Stage 3 README](../stage3-go2-training/README.md).

### Stage 4 — vision-conditioned locomotion ⏳

Not started. Would require depth-camera sensors in the MJCF, an extended
observation space, a CNN encoder, and heightfield terrain.

### Stage 5 — ROS2 deployment ⏳

Not started. No ROS2 package, node, or Unitree SDK2 binding exists.

### Stage 6 — sim-to-real 🔮

Not started; hardware-dependent.

### Engineering gaps

| Gap | Impact |
|---|---|
| ~~No tests~~ | **RESOLVED** — 93 tests, none needing policy weights |
| ~~No CI workflow~~ | **RESOLVED then removed** — local `make test` / `check_env.py` instead |
| No linter/formatter config | style drifts; unused imports survive (see R3) |
| No logging module | everything is `print()`; fine for scripts, awkward for long sweeps |
| No CLI argument parsing in exp1/2/3 | velocities, gaits and seeds are edited in source (Stage 3 scripts do have `argparse`) |
| No structured run metadata | CSVs record results but not git SHA, versions, or wall time |
| No dependency lockfile | `requirements.txt` has ranges, no hashes |
| ~~Root README truncated~~ | **RESOLVED** |

---

## 9. Suggested first contributions

Ordered by (value ÷ risk). Every one is small enough for a first PR.

Items 1, 2, 5, 6 and 11 from the original list are **done** — see the resolved
entries above. What remains:

**Tier 1 — safe, high value**

1. **Mark `04_build_obs_vector.py` superseded** with a header banner (R4).
   `tests/test_constants.py` now asserts it still *disagrees* with
   `harness.py`, but the file itself carries no warning.
2. **Fix the stale comments** in R5 and R8.
3. **Add a linter config** (ruff). The stray `turtle` import in R3 would have
   been caught automatically.

**Tier 2 — needs a judgement call**

4. **Add `argparse` to `exp1/2/3`** so sweeps are configurable without editing
   source. The Stage 3 scripts show the pattern.
5. **Record run metadata** (git SHA, package versions, wall-clock) into each CSV.
6. **Extract `go2_common.py`** for `07`, `08` and `harness.py` only (see §5).
   Now safer to attempt: the drift guard and contract tests will catch mistakes.

**Tier 3 — real research work**

7. **Complete the gait × velocity grid.** Experiment 2 tested three gaits at a
   single speed and found — surprisingly — that *pace* transfers better than
   trot. `EXPERIMENT_FINDINGS.md` flags the single-speed limitation itself.
   Filling the grid is ~90 trials against an existing harness: a genuinely
   publishable result reachable with code that already exists.
8. **Train a Stage 3 policy on GPU.** The pipeline is complete and verified;
   what is missing is compute.
9. **Port the Stage 3 env to MJX.** Contained to `env/go2_env.py`.

Start with #7 — it is real research you can run tonight on a CPU, and the
harness for it already exists.

---

## Related

- [ARCHITECTURE.md](ARCHITECTURE.md) — how the system fits together
- [FEATURES.md](FEATURES.md) — the same implemented/planned split as a matrix
- [SRS.md](SRS.md) · [TDD.md](TDD.md) — requirements and design
- `../stage2-go2-mujoco-inference/experiments/results/EXPERIMENT_FINDINGS.md` — the research findings
