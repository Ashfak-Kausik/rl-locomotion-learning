"""
Network architecture — and, more importantly, the CONTRACT.

Everything Stage 2 built depends on four numbers:

    OBS_DIM      70     one observation frame
    HISTORY_LEN  30     frames the adaptation module sees
    LATENT_DIM    2     environment latent inferred by the adaptation module
    ACTION_DIM   12     joint position deltas

    adaptation_module : (B, 30*70=2100)          -> (B, 2)
    body              : (B, 2100 + 2 = 2102)     -> (B, 12)

A policy trained here is only useful if it honours that contract exactly,
because then `stage2-go2-mujoco-inference/experiments/harness.py` can evaluate
it with zero changes — same scenes, same metrics, same figures, directly
comparable against the walk-these-ways baseline. That comparability is the
entire payoff of Stage 2's measurement work. Do not break it for convenience.

RMA (Rapid Motor Adaptation) in two phases
------------------------------------------
Phase 1  A *privileged* encoder sees the true environment parameters
         (friction, mass, motor strength) and compresses them to a latent.
         PPO trains {privileged encoder, body, critic} together.

Phase 2  The adaptation module learns, by supervised regression, to predict
         that same latent from proprioceptive HISTORY alone — because the
         deployed robot cannot measure its own friction coefficient.

Only `adaptation_module` and `body` are exported. The privileged encoder and
the critic are training scaffolding and are discarded.
"""

import torch
import torch.nn as nn

# ============================================================================
# THE CONTRACT — these values are fixed by Stage 2. Do not change them.
# ============================================================================
OBS_DIM = 70
HISTORY_LEN = 30
LATENT_DIM = 2
ACTION_DIM = 12

HISTORY_DIM = HISTORY_LEN * OBS_DIM          # 2100
BODY_INPUT_DIM = HISTORY_DIM + LATENT_DIM    # 2102

# Number of privileged environment parameters (see env/domain_rand.py).
PRIV_DIM = 8


def _mlp(sizes, activation=nn.ELU, output_activation=None):
    """Plain MLP. ELU is the walk-these-ways/legged_gym convention."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        is_last = i == len(sizes) - 2
        if not is_last:
            layers.append(activation())
        elif output_activation is not None:
            layers.append(output_activation())
    return nn.Sequential(*layers)


# ============================================================================
# Exported networks — these two, and only these two, become .jit files
# ============================================================================
class AdaptationModule(nn.Module):
    """
    (B, 2100) -> (B, 2)

    The student. Infers what kind of world the robot is in from how the robot
    has been responding over the last 0.6 s (30 frames at 50 Hz).
    """

    def __init__(self, hidden=(256, 128)):
        super().__init__()
        self.net = _mlp([HISTORY_DIM, *hidden, LATENT_DIM])

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        return self.net(obs_history)


class Body(nn.Module):
    """
    (B, 2102) -> (B, 12)

    The actor. Consumes [flattened history | environment latent] and emits 12
    joint-position deltas, which Stage 2 scales by ACTION_SCALE and adds to
    DEFAULT_JOINT_POS before the PD controller turns them into torques.

    Outputs are unbounded on purpose: the training-time Gaussian samples around
    them, and deployment uses the mean directly.
    """

    def __init__(self, hidden=(512, 256, 128)):
        super().__init__()
        self.net = _mlp([BODY_INPUT_DIM, *hidden, ACTION_DIM])

    def forward(self, body_input: torch.Tensor) -> torch.Tensor:
        return self.net(body_input)


# ============================================================================
# Training-only networks — discarded at export
# ============================================================================
class PrivilegedEncoder(nn.Module):
    """
    (B, PRIV_DIM) -> (B, 2)

    Phase-1 teacher. Sees ground-truth simulation parameters that a real robot
    could never measure. Its output is the regression target for the
    adaptation module in phase 2.
    """

    def __init__(self, hidden=(64, 32)):
        super().__init__()
        self.net = _mlp([PRIV_DIM, *hidden, LATENT_DIM])

    def forward(self, priv: torch.Tensor) -> torch.Tensor:
        return self.net(priv)


class Critic(nn.Module):
    """
    (B, 2102) -> (B, 1)

    PPO value function. Sees the same input as the body network. Never
    exported — inference has no use for a value estimate.
    """

    def __init__(self, hidden=(512, 256, 128)):
        super().__init__()
        self.net = _mlp([BODY_INPUT_DIM, *hidden, 1])

    def forward(self, body_input: torch.Tensor) -> torch.Tensor:
        return self.net(body_input).squeeze(-1)


# ============================================================================
# The agent — ties them together for training
# ============================================================================
class Go2Agent(nn.Module):
    """
    Everything PPO needs. `init_log_std` starts exploration wide; the learned
    log-std shrinks as the policy commits, exactly the `train/std` curve
    Stage 1 taught you to read.
    """

    def __init__(self, init_log_std=-1.0):
        super().__init__()
        self.body = Body()
        self.critic = Critic()
        self.privileged_encoder = PrivilegedEncoder()
        self.adaptation_module = AdaptationModule()
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), init_log_std))

    # -- phase 1: latent comes from the privileged encoder --------------
    def act(self, obs_history, priv, deterministic=False):
        """Returns (action, log_prob, entropy, value)."""
        latent = self.privileged_encoder(priv)
        body_input = torch.cat([obs_history, latent], dim=-1)
        mean = self.body(body_input)
        value = self.critic(body_input)

        if deterministic:
            return mean, None, None, value

        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        return (action,
                dist.log_prob(action).sum(-1),
                dist.entropy().sum(-1),
                value)

    def evaluate_actions(self, obs_history, priv, actions):
        """Re-evaluate stored actions under the current policy (PPO update)."""
        latent = self.privileged_encoder(priv)
        body_input = torch.cat([obs_history, latent], dim=-1)
        mean = self.body(body_input)
        value = self.critic(body_input)

        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        return (dist.log_prob(actions).sum(-1),
                dist.entropy().sum(-1),
                value)

    # -- phase 2 / deployment: latent comes from history ----------------
    def act_deployed(self, obs_history):
        """
        The deployment path — exactly what Stage 2's harness will do.
        Uses the adaptation module, never the privileged encoder.
        """
        latent = self.adaptation_module(obs_history)
        return self.body(torch.cat([obs_history, latent], dim=-1))


def assert_contract(adaptation_module, body, device="cpu"):
    """
    Verify a pair of networks honours the Stage 2 contract.

    Called by export.py before writing any file, and by the test suite. This is
    the guard that keeps Stage 3's output evaluable by Stage 2's harness.
    """
    adaptation_module = adaptation_module.to(device).eval()
    body = body.to(device).eval()

    with torch.no_grad():
        history = torch.zeros(1, HISTORY_DIM, device=device)
        latent = adaptation_module(history)
        if tuple(latent.shape) != (1, LATENT_DIM):
            raise ValueError(
                f"adaptation_module: expected output (1, {LATENT_DIM}), "
                f"got {tuple(latent.shape)}"
            )

        body_input = torch.cat([history, latent], dim=1)
        if tuple(body_input.shape) != (1, BODY_INPUT_DIM):
            raise ValueError(
                f"body input: expected (1, {BODY_INPUT_DIM}), "
                f"got {tuple(body_input.shape)}"
            )

        action = body(body_input)
        if tuple(action.shape) != (1, ACTION_DIM):
            raise ValueError(
                f"body: expected output (1, {ACTION_DIM}), "
                f"got {tuple(action.shape)}"
            )
        if not torch.isfinite(action).all():
            raise ValueError("body produced non-finite actions on zero input")

    return True
