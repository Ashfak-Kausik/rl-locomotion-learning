# Training speed

How to get Stage 3 PPO throughput as high as this codebase allows on a given
machine — and what will **not** help until the physics backend changes.

**Companion tools:** `scripts/hw_profile.py`, `stage3-go2-training/tune_profiles.py`,
`train.py --rollout-workers`, `make gpu` / `make hw-check`.

---

## 1. Where the time goes (measured)

On the reference host (Pop!\_OS, **i7-9700K 8c/8t no SMT**, **RTX 3050 8 GB**):

| Phase | What runs | Share of wall time | Device |
|---|---|---|---|
| **Rollout collection** | `mujoco.mj_step` × DECIMATION × num_envs | **~80–90%** | **CPU** |
| PPO update | actor/critic backward on history tensors | ~10–20% | GPU |
| Adaptation (phase 2) | supervised fit of the 2-dim latent | small, once at the end | GPU |

Evidence:

- GPU utilisation during a live run sits around **10–20%** — the card is waiting
  on the CPU to finish rollouts.
- End-to-end training: ~600 sps (CPU torch) → ~1,050–1,170 sps (CUDA) ≈ **1.8×**.
- Same card on a 4096³ matmul alone: **7×**. Rendering EGL vs OSMesa: **47×**.
- So “buy a bigger GPU” on this stack buys almost nothing for *training*
  throughput until physics moves off the CPU.

The rollout loop used to step envs **one after another** in Python. Each
`Go2Env` owns its own `MjModel` / `MjData`, and `mj_step` **releases the GIL**.
A thread pool over envs therefore scales with cores:

| | Wall for 8 envs × 200 `mj_step`s |
|---|---|
| Serial | 0.075 s |
| 8 threads | 0.020 s |
| **Speedup** | **~3.7×** (physics only, this CPU) |

That is the single largest win available without an MJX port.

---

## 2. What to turn on (checklist)

### Reward / curriculum (v8+ defaults)

After multigait_v4–v7, the trainer defaults include:

| Fix | What it does |
|---|---|
| `alive = 0` | No free survival bonus; fall penalty (-10) still punishes drops |
| `low_body_height` | Linear penalty below 0.25 m — blocks floor-dig / crouch |
| Curriculum movement gate | Promote only if **return AND** (vx ≥ 0.15 m/s OR dist ≥ 1.5 m/ep) |
| `--flat-bootstrap` | Flat trot, no curriculum/domain rand until walking works |

Always `diagnose.py` after a probe before approving a long budget.

### Always

```bash
source .venv/bin/activate
./scripts/setup_env.sh --gpu auto     # matching torch wheel
make hw-profile                       # record the machine
python scripts/check_env.py
python stage3-go2-training/train.py --smoke
```

### Fast path for a real run (current code)

```bash
# rollout_workers defaults to 0 = auto = min(num_envs, cpu_count)
python stage3-go2-training/train.py \
  --run-name my_run \
  --device auto \
  --timesteps 15000000
```

| Knob | Default | Notes |
|---|---|---|
| `--rollout-workers 0` | **auto** | `min(num_envs, os.cpu_count())` — use this |
| `--rollout-workers 1` | serial | old behaviour; only for debugging |
| `--device auto` | CUDA/XPU/MPS if usable | never silent CPU fallback |
| `--tune-profile …` | auto from VRAM | see `python stage3-go2-training/tune_profiles.py --all` |
| `--num-envs N` | from profile | **do not** raise past ~4× physical cores on this stack |

### Host hygiene (free, cumulative)

- Close browsers / other MuJoCo viewers while training (they fight for CPU).
- Prefer `MUJOCO_GL=osmesa` or unset for training — you are not rendering.
- Do not encode demo videos on the same box mid-run.
- After an NVIDIA driver upgrade: **reboot** before measuring (`make gpu`).
- Performance CPU governor if your distro defaults to powersave.

### Budget hygiene (often bigger than sps)

A 15M-step run that stands still for 13M steps is slower than a 2M-step run
you diagnose early:

```bash
# probe — ~3 min at 1k sps
python stage3-go2-training/train.py --run-name probe --timesteps 200000

# as soon as checkpoint_best looks “good” by return:
python stage3-go2-training/diagnose.py --run probe
# if VERDICT: STANDS STILL → stop, fix reward, new --run-name. Do not burn 15M.
```

`checkpoint_best.pt` freezing for hours while `latest` keeps updating is a
stop signal, not a reason to wait for the budget.

---

## 3. What does **not** help (on this architecture)

| Idea | Why it fails here |
|---|---|
| Raise `num_envs` to 64/128 on an 8-core CPU | Rollouts are sequential-per-worker; past ~4× cores you add latency, not sps. `tune_profiles.select()` already caps by core count. |
| Raise `minibatch_size` alone | Speeds the GPU slice (~10–20% of wall). Easy to break the rule `minibatch < num_envs * rollout_steps` and silently full-batch. |
| Enable TF32 | Faster matmuls, but put a fake floor under `approx_kl` that made the KL guard reject **every** minibatch for millions of steps (`multigait_v4`). Stay off until re-measured. |
| “Just use a 4090” | Same MuJoCo CPU bottleneck unless the host also has many cores **or** you port to MJX. |
| Shorter `episode_seconds` as a speed hack | Changes the MDP (truncation / credit assignment). Use only for smoke tests. |

---

## 4. Profiles on this machine

Committed profile: `hardware/profiles/pop-os.json`.

Active tune profile when CUDA is up: **`rtx3050-8gb`**
(`num_envs=32`, `rollout_steps=64`, `minibatch_size=512`, `tf32=False`,
`torch_threads=8`).

Expected ballpark after threaded rollouts (auto workers):

| Config | Rough sps | 15M-step wall |
|---|---|---|
| CPU, serial rollouts | ~600 | ~7 h |
| CUDA, serial rollouts (v5/v6 era) | ~1,050–1,170 | ~3.5–4 h |
| CUDA + threaded rollouts (default now) | **target ~2–3k+** | **~1.5–2 h** (measure; update this table) |

Re-measure after any change:

```bash
python stage3-go2-training/train.py \
  --run-name speed_probe --timesteps 200000 --device auto
# read the `sps` column; then:
python scripts/hw_profile.py --save
# if sustained sps moved a profile’s rationale, patch tune_profiles.py
```

---

## 5. The ceiling: MJX / batched physics

Threaded MuJoCo is still **one robot per thread**. The next order-of-magnitude
is GPU-batched physics:

- Only `env/go2_env.py` is backend-specific.
- Port target: MJX + `jax.vmap` (see `stage3-go2-training/README.md` § MJX).
- Then a 3050’s VRAM and matmul throughput finally dominate wall time.

Until that port lands, treat **rollout workers + early diagnose** as the
practical speed toolkit.

---

## 6. Intel Arc / AMD / Apple

Install the right wheel first ([DEPENDENCIES.md §4.1](DEPENDENCIES.md#41-gpu-flavours--nvidia-intel-arc-amd-apple)).
Rollout threading helps **every** vendor equally (it is CPU-side). GPU
profiles for Arc/ROCm/MPS are unbenchmarked starting points — run
`speed_probe`, then PR measured numbers into `tune_profiles.py`.

---

## Related

- [`DEPENDENCIES.md` §4.1](DEPENDENCIES.md#41-gpu-flavours--nvidia-intel-arc-amd-apple) — torch flavours
- [`NATIVE-SETUP.md`](NATIVE-SETUP.md) — this host’s package inventory
- `stage3-go2-training/tune_profiles.py` — batch sizes by backend/VRAM
- `stage3-go2-training/diagnose.py` — don’t spend budget on standers
