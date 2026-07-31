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
        "tracking_lin_vel": 1.5,
        # 0.5 -> 0.35: empirically, standing perfectly still scores this term
        # near its max (yaw rate is trivially ~0), while an actual trot's
        # natural small yaw-rate oscillation scores meaningfully lower
        # (measured: 0.50/step standing vs 0.28/step walking, same env, real
        # walk-these-ways policy). That shrinks the reward margin that should
        # be pulling early-training exploration toward walking. Total reward
        # was still higher walking than standing either way (1.29 vs 1.00/
        # step) -- this is not "stand-still is the local optimum", it's
        # "the margin is smaller than it should be". See the diagnostic run
        # referenced in EXPERIMENT_FINDINGS.md / stage3 README.
        "tracking_ang_vel": 0.35,
        "alive": 0.5,
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
    domain_rand: bool = True
    fall_penalty: float = -10.0
    tracking_sigma: float = 0.25
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
    target_kl: float = 0.02      # early-stop an update; see Stage 1 findings
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
    gpu_minibatch_size: int = 2048
    tf32: bool = True            # Ampere+ tensor cores for fp32 matmuls

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
