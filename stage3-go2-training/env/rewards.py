"""
Reward terms for Go2 locomotion.

Design intent, in one line: **reward tracking the commanded velocity, and
penalise everything that makes a gait ugly, fragile or expensive.**

Two choices here are driven directly by Stage 2's measurements rather than by
convention:

  * `tracking_lin_vel` uses an exponential kernel, not a negative absolute
    error. Exp 1 found the baseline policy systematically UNDER-tracks
    (37-55% of command). A squared-error penalty saturates and stops pushing
    once the error is large; the exponential kernel keeps a usable gradient
    all the way in, which is what you want when the failure mode is
    "consistently too slow".

  * `lateral_drift` gets its own term. Exp 1 F1.3 found drift is non-monotonic
    in commanded speed, with a pronounced minimum at 1.0 m/s — evidence the
    baseline never had drift penalised directly.

Every function returns a scalar for one timestep. Weights live in config.py so
reward shaping can be swept without touching this file.
"""

import numpy as np


def tracking_lin_vel(cmd_vx, cmd_vy, actual_vx, actual_vy, sigma=0.25):
    """
    Exponential kernel on planar velocity error. Range (0, 1].

    sigma sets how forgiving it is: 0.25 means an error of 0.25 m/s scores
    e^-1 = 0.37. Smaller sigma is stricter.
    """
    err = (cmd_vx - actual_vx) ** 2 + (cmd_vy - actual_vy) ** 2
    return float(np.exp(-err / (sigma ** 2)))


def tracking_ang_vel(cmd_yaw, actual_yaw, sigma=0.25):
    """Same kernel, applied to yaw rate."""
    return float(np.exp(-((cmd_yaw - actual_yaw) ** 2) / (sigma ** 2)))


def lateral_drift(actual_vy):
    """Penalise sideways motion. Negative; scaled by the config weight."""
    return -float(abs(actual_vy))


def vertical_velocity(vz):
    """Penalise bouncing. Squared, so big hops hurt disproportionately."""
    return -float(vz ** 2)


def body_orientation(projected_gravity):
    """
    Penalise pitch and roll. Upright means projected gravity is [0, 0, -1],
    so any x/y component is unwanted tilt. Yaw is deliberately not penalised —
    heading is commanded, not fixed.
    """
    return -float(projected_gravity[0] ** 2 + projected_gravity[1] ** 2)


def body_height(height, target=0.30):
    """Hold the nominal standing height the policy was posed at."""
    return -float((height - target) ** 2)


def joint_torque(torques):
    """Energy proxy. Discourages stiff, high-current gaits."""
    return -float(np.sum(np.square(torques)))


def joint_velocity(joint_vels):
    """Discourages frantic leg motion."""
    return -float(np.sum(np.square(joint_vels)))


def joint_acceleration(joint_vels, prev_joint_vels, dt):
    """Smoothness. Large accelerations are what destroy real gearboxes."""
    accel = (joint_vels - prev_joint_vels) / max(dt, 1e-6)
    return -float(np.sum(np.square(accel)))


def action_rate(action, prev_action):
    """
    Penalise how fast the policy changes its mind. The single most effective
    term for suppressing the high-frequency chatter that ruins sim-to-real.
    """
    return -float(np.sum(np.square(action - prev_action)))


def action_magnitude(action):
    """Keep deltas near the default pose; discourages extreme postures."""
    return -float(np.sum(np.square(action)))


def joint_limits(joint_pos, lower, upper, soft_ratio=0.9):
    """
    Penalise approaching a joint's mechanical limit. Hitting a hard stop in
    simulation is free; on hardware it is damage.
    """
    mid = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower) * soft_ratio
    excess = np.clip(np.abs(joint_pos - mid) - half, 0.0, None)
    return -float(np.sum(excess))


def alive():
    """Constant bonus per surviving step — offsets the penalty terms so that
    standing still is better than falling over immediately."""
    return 1.0


def compute_total(terms: dict, weights: dict):
    """
    Weighted sum. Returns (total, per_term_contribution).

    The breakdown is returned because reward debugging is impossible without
    it: a policy that refuses to move is almost always one term dominating,
    and you can only see that per-term.
    """
    contributions = {
        name: weights.get(name, 0.0) * value
        for name, value in terms.items()
    }
    return float(sum(contributions.values())), contributions
