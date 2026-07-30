"""
Training configuration.

One dataclass, all defaults in one place, everything overridable from the CLI.
Reward weights in particular are the thing you will sweep most, so they live in
a plain dict rather than being scattered through the env.
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
        "tracking_ang_vel": 0.5,
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
    device: str = "cpu"          # "cuda" when a GPU is available
    out_dir: str = "runs"

    # --- environment ---------------------------------------------------
    episode_seconds: float = 10.0
    gait: str = "trot"
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
