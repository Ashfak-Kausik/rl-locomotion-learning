"""
Training configuration.

One dataclass, all defaults in one place, everything overridable from the CLI.
Reward weights in particular are the thing you will sweep most, so they live in
a plain dict rather than being scattered through the env.

Device handling lives here too: `resolve_device()` turns "auto" into a real
device and explains itself, and `tune_for_device()` scales the PPO update to
match whichever one you got.
"""

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
        "tracking_lin_vel": 2.0,
        # Linear and non-saturating: the exponential kernel above is flat
        # far from target, so a stationary robot cannot feel the gradient.
        # This one always pays to move faster. See rewards.forward_progress.
        "forward_progress": 1.0,
        # 0.35 -> 0.2. Standing scores this near its maximum (yaw rate is
        # trivially ~0) while a real trot's natural yaw oscillation scores
        # lower -- measured 0.50/step standing vs 0.28/step walking. It was
        # the second-largest reward source for doing nothing.
        "tracking_ang_vel": 0.2,
        # 0.5 -> 0.15. This is a pure participation trophy: paid every step
        # for not having fallen over, and it was 67% of the standing
        # policy's entire reward. It still needs to be positive so that
        # falling is worse than surviving, but it must not be competitive
        # with actually moving.
        "alive": 0.15,
        # --- stability -------------------------------------------------
        "lateral_drift": 0.5,      # Exp 1 F1.3: baseline never penalised this
        "vertical_velocity": 0.5,
        "body_orientation": 1.0,
        "body_height": 5.0,
        # --- smoothness / energy ---------------------------------------
        "action_rate": 0.01,
        "action_magnitude": 0.001,
        "joint_torque": 0.0002,
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
    # 0.25 -> 0.4. sigma sets how far from target the exponential kernel
    # still has usable gradient. At 0.25, a stationary robot commanded to
    # 0.5 m/s scores 0.018 with gradient 0.29 -- effectively flat, so it
    # cannot tell that moving would help. At 0.4 the same state scores 0.210
    # with gradient 1.31, a 4.5x stronger learning signal exactly where the
    # policy was stuck.
    tracking_sigma: float = 0.4
    reward_weights: dict = field(default_factory=default_reward_weights)

    # --- curriculum ----------------------------------------------------
    curriculum: bool = True
    promote_threshold: float = 0.75
    demote_threshold: float = 0.30
    curriculum_window: int = 20

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
    # Scale the PPO update up when a GPU is present. The bottleneck differs
    # by device: on CPU it is MuJoCo stepping, on GPU it is kernel-launch
    # overhead, so bigger minibatches and more envs pay off there and cost
    # you on CPU. Applied by tune_for_device().
    gpu_num_envs: int = 32
    gpu_rollout_steps: int = 64
    # steps_per_update = gpu_num_envs * gpu_rollout_steps = 2048. This MUST
    # stay strictly smaller than that, or "minibatch" silently becomes the
    # entire batch and update_epochs collapses into repeated full-batch
    # gradient steps with zero stochastic-minibatch diversity between them --
    # exactly the PPO anti-pattern that produces sustained high clip-fraction
    # and KL blowups. Found by a real run: clip_fraction sat at 0.3-0.7 and
    # return oscillated wildly (14 -> 673 -> -181 -> 456 ...) for its entire
    # duration when this was 2048 == batch size. 512 gives 4 minibatches/
    # epoch, standard PPO practice, at negligible extra kernel-launch cost on
    # a GPU this small.
    gpu_minibatch_size: int = 512
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

    available = torch.cuda.is_available()
    built_with_cuda = torch.version.cuda is not None

    if requested == "cpu":
        say(f"using CPU by request ({torch.get_num_threads()} threads)")
        return torch.device("cpu")

    if available:
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

    return torch.device("cpu")


def tune_for_device(cfg, device):
    """
    Scale the PPO update to the device.

    CPU: MuJoCo stepping dominates, so keep envs and minibatches small.
    GPU: kernel-launch overhead dominates, so batch aggressively — a 2048-row
    minibatch costs a 3050 barely more than a 128-row one.

    Mutates and returns cfg. Only touches values the user did not override.
    """
    import torch

    if device.type != "cuda":
        return cfg

    cfg.num_envs = cfg.gpu_num_envs
    cfg.rollout_steps = cfg.gpu_rollout_steps
    cfg.minibatch_size = cfg.gpu_minibatch_size

    if cfg.tf32:
        # TF32 matmuls on Ampere+: ~same accuracy for RL, materially faster.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    print(f"[device] tuned for GPU: num_envs={cfg.num_envs} "
          f"rollout_steps={cfg.rollout_steps} "
          f"minibatch={cfg.minibatch_size} tf32={cfg.tf32}")
    return cfg
