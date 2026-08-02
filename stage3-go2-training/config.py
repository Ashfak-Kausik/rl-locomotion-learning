"""
Training configuration.

One dataclass, all defaults in one place, everything overridable from the CLI.
Reward weights in particular are the thing you will sweep most, so they live in
a plain dict rather than being scattered through the env.

Device handling lives here too: `resolve_device()` turns "auto" into a real
device and explains itself, and `tune_for_device()` scales the PPO update to
match whichever one you got. The per-machine numbers it applies live in
`tune_profiles.py`.
"""

import os
from dataclasses import dataclass, field


def default_reward_weights():
    """
    Starting weights. Positive terms reward, negative terms penalise.

    Rough magnitude discipline: the two tracking terms plus `alive` dominate
    (they are the objective), and every penalty is scaled so its typical
    per-step contribution is well under 1.0. A penalty that can outweigh
    `tracking_lin_vel` produces a policy that stands perfectly still — which is
    the single most common locomotion reward-shaping failure.
    """
    return {
        # --- objective -------------------------------------------------
        # Rebalanced after multigait_v4 converged to standing perfectly
        # still. Its measured per-step breakdown was:
        #     alive             0.500   <- 67% of reward, free for standing
        #     tracking_ang_vel  0.317   <- 42%, maximised by NOT turning
        #     tracking_lin_vel  0.029   <- 3.8%, the actual objective
        # i.e. 0.82/step for standing still versus 0.03 for the thing we
        # want. Standing was not a bug, it was the rational optimum.
        "tracking_lin_vel": 2.5,
        # 2.5 -> 5.0 after v8_flat 200k probe: objective mass was 42% but
        # vx still 0.009 m/s — forward_progress at half-command should
        # out-earn standing (~0.05/step) by an order of magnitude.
        "forward_progress": 5.0,
        # Straight-line locomotion, not drift. v8/v9 walked (vx ~0.3 m/s)
        # but spiralled: body vx was fine while the world path curved back,
        # so distance_m stayed ~0.5 m and curriculum never promoted.
        "heading_hold": 2.0,
        # 0.2 -> 0.05. Measured on v6 best: 0.195/step (38% of reward) while
        # stationary — yaw rate is trivially ~0 when you do not move. Keep a
        # trickle so commanded yaw turns still matter; do not subsidise idle.
        "tracking_ang_vel": 0.05,
        # 0.15 -> 0.05. Same story: pure survival bonus, 29% of v6 standing
        # reward. Fall penalty (-10) still makes dying worse than surviving.
        # v7 still stood still with 0.05 — removed entirely; only fall_penalty
        # offsets the negative stability terms now.
        "alive": 0.0,
        # --- stability -------------------------------------------------
        "lateral_drift": 0.5,      # Exp 1 F1.3: baseline never penalised this
        "vertical_velocity": 0.5,
        "body_orientation": 1.0,
        # 20 -> 8. v8_flat best stood at h=0.258 m; squared pull at weight 20
        # cost -0.036/step — comparable to joint_torque and fighting the
        # bounce of a nascent gait. low_body_height still blocks crouch/dig.
        "body_height": 8.0,
        # Linear floor at 0.25 m — kills crouch / floor-dig (v5–v7). At h=0.21
        # the gap is 0.04 m -> -0.4/step at weight 10, comparable to idle hacks.
        "low_body_height": 10.0,
        # --- smoothness / energy ---------------------------------------
        "action_rate": 0.01,
        "action_magnitude": 0.001,
        "joint_torque": 0.0001,
        "joint_velocity": 0.001,
        "joint_acceleration": 2.5e-7,
    }


@dataclass
class Config:
    # --- run -----------------------------------------------------------
    run_name: str = "go2_ppo"
    seed: int = 0
    device: str = "auto"         # "auto" | "cpu" | "cuda" — see resolve_device
    out_dir: str = "runs"

    # --- environment ---------------------------------------------------
    episode_seconds: float = 10.0
    gait: str = "trot"           # used when randomize_gait is False
    # Sample a gait per episode from this pool instead of training on one
    # fixed gait. "All types of move" as a single policy, not three separate
    # ones — same approach walk-these-ways itself used (the gait command is
    # already part of the 70-dim observation contract; nothing to add).
    randomize_gait: bool = True
    gait_pool: tuple = ("trot", "pace", "bound")
    # Hard bound on policy output before it becomes a joint target. See the
    # comment in go2_env.step -- an effectively unbounded action range is
    # what let multigait_v3 diverge. +-5 => +-1.25 rad of joint offset.
    action_clip: float = 5.0
    domain_rand: bool = True
    fall_penalty: float = -10.0
    # Reverted 0.4 -> 0.25 after multigait_v5. Widening sigma was meant to
    # give a stationary robot a usable gradient toward 0.5 m/s, but it also
    # multiplied the standing-still payout by ~10x (exp(-0.5^2/0.4^2)=0.21
    # vs exp(-0.5^2/0.25^2)=0.018). Measured on v5's "best" checkpoint:
    # tracking_lin_vel alone was 45% of reward mass while body vx was 0.01
    # m/s — standing WAS the optimum again. Keep sigma tight; rely on
    # forward_progress (linear, non-saturating) for the far-from-target
    # gradient instead.
    tracking_sigma: float = 0.25
    reward_weights: dict = field(default_factory=default_reward_weights)

    # --- curriculum ----------------------------------------------------
    curriculum: bool = True
    promote_threshold: float = 0.75
    demote_threshold: float = 0.30
    curriculum_window: int = 20
    # Must show real locomotion before leaving flat-slow (see curriculum.py).
    promote_min_body_vx: float = 0.15
    # v8/v9 moved (vx ~0.3 m/s) but curved, so displacement stayed < 1.5 m
    # and curriculum never left flat-slow. Path distance (see go2_env) or a
    # lower displacement floor is what a curvy-but-moving gait can pass.
    promote_min_distance_m: float = 0.8
    # Crouch / floor height penalty activates below this body height (metres).
    low_height_min: float = 0.25

    # --- PPO (phase 1) -------------------------------------------------
    total_timesteps: int = 5_000_000
    num_envs: int = 8            # sequential MuJoCo instances
    rollout_steps: int = 64      # per env, per update
    minibatch_size: int = 128
    update_epochs: int = 5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    learning_rate: float = 3e-4
    # Early-stop an update; see Stage 1 findings. Checked per-minibatch since
    # the divergence-guard fix (train.py's ppo_update), which is stricter
    # than the common per-epoch-mean check -- confirmed by the new
    # kl/kl_max/n_rejected diagnostics: at gpu_minibatch_size=512, 0.02
    # rejected roughly half of every epoch's minibatches from update 1
    # onward on a fresh network, meaning most of each rollout's learning
    # signal was being discarded every single update, not just as a rare
    # safety net. Loosened to 0.035 -- still well below values (0.05-0.1)
    # common in other PPO implementations -- to let more of a healthy epoch
    # actually apply, while still rejecting genuinely large jumps.
    target_kl: float = 0.035
    init_log_std: float = -1.0

    # --- adaptation module (phase 2) -----------------------------------
    adapt_steps: int = 100_000
    adapt_batch_size: int = 256
    adapt_learning_rate: float = 1e-3

    # --- checkpointing -------------------------------------------------
    # Free-tier GPU sessions get killed without warning; checkpoint often.
    save_every_updates: int = 20
    log_every_updates: int = 1

    # --- GPU -----------------------------------------------------------
    # num_envs / rollout_steps / minibatch_size are overwritten by
    # tune_for_device() using the machine-specific numbers in
    # tune_profiles.py. They used to be hardcoded here as gpu_num_envs=32 /
    # gpu_minibatch_size=512, which are the MEASURED values for an RTX 3050
    # + 8-core i7-9700K -- and were then applied unchanged to every other
    # CUDA device, including 4 GB cards that cannot fit them and Arc cards
    # that never got here at all. Set by tune_for_device() for the record.
    tune_profile: str = ""
    # Explicit CLI overrides, honoured by tune_for_device(). Before these
    # existed, `--num-envs 4` was accepted, printed, and then silently
    # overwritten by the GPU tuning block a few lines later -- the flag did
    # nothing at all on a GPU box.
    num_envs_override: int = 0
    minibatch_size_override: int = 0
    # Parallelise env.step across CPU cores during rollout collection.
    # 0 = auto (min(num_envs, os.cpu_count())); 1 = serial; N = N threads.
    # Safe because each Go2Env owns its own MjModel/MjData and mj_step
    # releases the GIL. See docs/TRAINING-SPEED.md.
    rollout_workers: int = 0
    # The one sizing rule that must never be broken, wherever the numbers
    # come from: minibatch_size MUST stay strictly smaller than
    # num_envs * rollout_steps, or "minibatch" silently becomes the entire
    # batch and update_epochs collapses into repeated full-batch gradient
    # steps with zero stochastic-minibatch diversity between them -- exactly
    # the PPO anti-pattern that produces sustained high clip-fraction and KL
    # blowups. Found by a real run: clip_fraction sat at 0.3-0.7 and return
    # oscillated wildly (14 -> 673 -> -181 -> 456 ...) for its entire duration
    # when minibatch_size == batch size. TuneProfile.validate() now enforces
    # this for every profile, so the mistake cannot be reintroduced.
    #
    # TF32 is OFF by default despite being faster on Ampere+. Measured: it
    # makes the FIRST minibatch of a PPO update report approx_kl = 0.0014
    # (max |dlogp| = 0.30 on individual samples) where the exact answer is
    # 0.0 -- same network, same data, ratio must be 1. The cause is
    # batch-size-dependent kernel selection: rollouts are collected at
    # batch=num_envs (32) and re-evaluated at batch=minibatch_size (512),
    # and reduced-mantissa matmuls round differently. With tf32=False the
    # same measurement gives exactly 0.000000.
    #
    # That numerical floor is not a policy change, but the KL guard cannot
    # tell the difference. Once it exceeds target_kl the guard rejects every
    # minibatch: multigait_v4 logged `rej 1/1` for its last ~10M steps and
    # applied ZERO gradients while appearing to train normally.
    tf32: bool = False

    def describe(self):
        lines = ["Config:"]
        for key, value in self.__dict__.items():
            if key == "reward_weights":
                lines.append("  reward_weights:")
                for name, weight in value.items():
                    lines.append(f"      {name:22s} {weight}")
            else:
                lines.append(f"  {key:22s} {value}")
        return "\n".join(lines)


# ===========================================================================
# Device selection
# ===========================================================================
def resolve_device(requested="auto", verbose=True):
    """
    Turn a device request into a real torch device, and say why.

    Silent CPU fallback is the failure mode to avoid: you ask for cuda, get
    cpu, and only notice three hours later when the run is 40x slower than
    expected. This is loud about what it picked and what went wrong.

    Returns a torch.device.
    """
    import torch

    def say(msg):
        if verbose:
            print(f"[device] {msg}")

    cuda_ok = torch.cuda.is_available()
    xpu_ok = bool(getattr(torch, "xpu", None) and torch.xpu.is_available())
    mps_ok = bool(getattr(torch.backends, "mps", None)
                  and torch.backends.mps.is_available())
    built_with_cuda = torch.version.cuda is not None

    if requested == "cpu":
        say(f"using CPU by request ({torch.get_num_threads()} threads)")
        return torch.device("cpu")

    # Explicit non-CUDA accelerators. Intel Arc in particular used to be
    # invisible here: `torch.cuda.is_available()` is False on an Arc box, so
    # every Arc contributor silently got CPU with no warning at all.
    if requested == "xpu" or (requested == "auto" and xpu_ok and not cuda_ok):
        if xpu_ok:
            say(f"using XPU: {torch.xpu.get_device_name(0)}")
            return torch.device("xpu")
        say("WARNING: --device xpu requested but torch.xpu is unavailable. "
            "Intel Arc needs the XPU wheel:\n"
            "           ./scripts/setup_env.sh --xpu")
        say("falling back to CPU — this will be MUCH slower")
        return torch.device("cpu")

    if requested == "mps" or (requested == "auto" and mps_ok and not cuda_ok):
        if mps_ok:
            say("using MPS (Apple Silicon)")
            return torch.device("mps")
        say("WARNING: --device mps requested but unavailable")
        return torch.device("cpu")

    if cuda_ok:
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        cap = torch.cuda.get_device_capability(idx)
        total = torch.cuda.get_device_properties(idx).total_memory / 1e9
        say(f"using CUDA: {name} (sm_{cap[0]}{cap[1]}, {total:.1f} GB, "
            f"CUDA {torch.version.cuda})")
        return torch.device("cuda")

    # Asked for (or would accept) CUDA but it is not usable — diagnose.
    if not built_with_cuda:
        reason = ("this torch is a CPU-only build. Reinstall with:\n"
                  "           pip install torch --index-url "
                  "https://download.pytorch.org/whl/cu130")
    else:
        reason = ("torch has CUDA support but no usable GPU was found. "
                  "Most often a driver/library version mismatch — run "
                  "`nvidia-smi`; if it errors, reboot.")

    if requested == "cuda":
        say(f"WARNING: --device cuda requested but unavailable. {reason}")
        say("falling back to CPU — this will be MUCH slower")
    else:
        say(f"no GPU available, using CPU. {reason}")
        # A GPU that exists but is unusable is worth naming explicitly —
        # otherwise the only symptom is a run that is 40x slower than
        # expected and nobody notices for three hours.
        _warn_about_unusable_gpu(say)

    return torch.device("cpu")


def _warn_about_unusable_gpu(say):
    """If lspci sees a GPU that torch cannot use, say so by vendor."""
    import shutil
    import subprocess
    if not shutil.which("lspci"):
        return
    try:
        out = subprocess.run(["lspci"], capture_output=True, text=True,
                             timeout=10).stdout
    except Exception:
        return
    lines = [ln for ln in out.splitlines()
             if any(k in ln for k in ("VGA", "3D controller", "Display"))]
    if any("Intel" in ln and "Arc" in ln for ln in lines):
        say("NOTE: an Intel Arc GPU is present but torch cannot use it. "
            "Install the XPU wheel: ./scripts/setup_env.sh --xpu")
    elif any("NVIDIA" in ln for ln in lines):
        say("NOTE: an NVIDIA GPU is present but torch cannot use it. "
            "Run `python scripts/gpu_check.py` for a diagnosis.")


def tune_for_device(cfg, device, profile_name=None):
    """
    Scale the PPO update to the machine.

    CPU: MuJoCo stepping dominates, so keep envs and minibatches small.
    GPU: kernel-launch overhead dominates, so batch aggressively — a 2048-row
    minibatch costs a 3050 barely more than a 128-row one.

    The actual numbers live in `tune_profiles.py`, keyed on backend, VRAM and
    core count, because they were measured on one specific machine and are
    wrong on others. This used to apply the RTX 3050 numbers to any CUDA
    device, including 4 GB cards that cannot fit them.

    `profile_name` forces a specific profile (train.py --tune-profile).

    Mutates and returns cfg.
    """
    import torch

    import tune_profiles

    if profile_name:
        if profile_name not in tune_profiles.BY_NAME:
            raise SystemExit(
                f"unknown tuning profile {profile_name!r}. Available: "
                f"{', '.join(sorted(tune_profiles.BY_NAME))}")
        profile = tune_profiles.BY_NAME[profile_name]
        print(f"[device] tuning profile forced: {profile.name}")
    else:
        vram = name = None
        if device.type == "cuda" and torch.cuda.is_available():
            idx = torch.cuda.current_device()
            vram = torch.cuda.get_device_properties(idx).total_memory / 1e9
            name = torch.cuda.get_device_name(idx)
        elif device.type == "xpu" and getattr(torch, "xpu", None):
            name = torch.xpu.get_device_name(0)
            try:
                vram = torch.xpu.get_device_properties(0).total_memory / 1e9
            except Exception:
                pass
        profile = tune_profiles.select(
            backend=device.type, vram_gb=vram, gpu_name=name,
            logical_cores=os.cpu_count(),
        )

    problems = profile.validate()
    if problems:
        raise SystemExit(f"tuning profile {profile.name} is invalid: "
                         + "; ".join(problems))

    cfg.num_envs = profile.num_envs
    cfg.rollout_steps = profile.rollout_steps
    cfg.minibatch_size = profile.minibatch_size
    cfg.tf32 = profile.tf32
    cfg.tune_profile = profile.name

    # Explicit CLI overrides win over the profile, and re-derive the
    # minibatch so the "never full-batch" rule survives the override.
    if cfg.num_envs_override:
        cfg.num_envs = cfg.num_envs_override
        batch = cfg.num_envs * cfg.rollout_steps
        if cfg.minibatch_size >= batch:
            cfg.minibatch_size = max(32, batch // 4)
            print(f"[device] --num-envs {cfg.num_envs} shrank the batch to "
                  f"{batch}; minibatch reduced to {cfg.minibatch_size} to "
                  f"keep it a real minibatch")
    if cfg.minibatch_size_override:
        cfg.minibatch_size = cfg.minibatch_size_override

    batch = cfg.num_envs * cfg.rollout_steps
    if cfg.minibatch_size >= batch:
        raise SystemExit(
            f"minibatch_size {cfg.minibatch_size} >= batch {batch}. This "
            f"silently turns PPO into full-batch gradient descent and has "
            f"already cost this project one multi-hour run; refusing to "
            f"start.")

    if profile.torch_threads:
        torch.set_num_threads(profile.torch_threads)

    if cfg.tf32 and device.type == "cuda":
        # TF32 matmuls on Ampere+: ~same accuracy for RL, materially faster.
        # Off by default — see the tf32 comment on Config.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    tag = "measured" if profile.measured else "ESTIMATED, unbenchmarked"
    print(f"[device] tuning profile: {profile.name} ({tag})")
    print(f"[device]   num_envs={cfg.num_envs} "
          f"rollout_steps={cfg.rollout_steps} "
          f"minibatch={cfg.minibatch_size} "
          f"({profile.minibatches_per_epoch} minibatches/epoch) "
          f"tf32={cfg.tf32}")
    if not profile.measured:
        print("[device]   this profile is a conservative guess for your "
              "hardware. If it")
        print("[device]   works, record it: python scripts/hw_profile.py "
              "--save")
    return cfg
