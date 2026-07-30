"""
Domain randomisation — and the privileged parameter vector it produces.

Why this exists
---------------
Experiment 1 measured the cost of *not* doing this: the walk-these-ways policy
achieves only 37-55% of commanded velocity after Isaac Gym -> MuJoCo transfer,
because it was tuned around one simulator's contact dynamics. A policy trained
across a *distribution* of dynamics has no single set of physics to overfit to.

Each episode samples one set of parameters. Those same values, normalised to
roughly [-1, 1], become the `privileged` vector fed to the phase-1 encoder
(networks.PrivilegedEncoder). In phase 2 the adaptation module learns to infer
that vector's latent from proprioceptive history alone — which is the whole
point of RMA, since a real robot cannot measure its own friction coefficient.

Keep PRIV_DIM in networks.py equal to len(PARAM_SPECS).
"""

from dataclasses import dataclass

import numpy as np

# (name, low, high, nominal) — nominal is the un-randomised MuJoCo value.
PARAM_SPECS = [
    # Ground friction. The single biggest sim-to-sim divergence: MuJoCo's
    # elliptic friction cone behaves differently from Isaac Gym's pyramidal one.
    ("friction",        0.30, 1.50, 0.60),
    # Trunk mass (kg added/removed). Represents payload and battery variation.
    ("added_mass",     -1.50, 3.00, 0.00),
    # Multiplier on the PD position gain — motor strength varies with heat
    # and battery charge on real hardware.
    ("kp_scale",        0.80, 1.20, 1.00),
    # Multiplier on the PD damping gain.
    ("kd_scale",        0.80, 1.20, 1.00),
    # Joint damping multiplier — models wear and lubrication.
    ("joint_damping",   0.70, 1.40, 1.00),
    # Per-joint torque scale — models per-motor manufacturing spread.
    ("motor_strength",  0.85, 1.15, 1.00),
    # Control latency in policy steps (0-2 steps = 0-40 ms at 50 Hz).
    # Real robots always have some; simulators default to none.
    ("action_latency",  0.00, 2.00, 0.00),
    # Centre-of-mass x offset (m) — payload never sits exactly on centre.
    ("com_offset_x",   -0.05, 0.05, 0.00),
]

PARAM_NAMES = [name for name, _, _, _ in PARAM_SPECS]
PRIV_DIM = len(PARAM_SPECS)


@dataclass
class DomainParams:
    """One episode's sampled dynamics."""

    friction: float
    added_mass: float
    kp_scale: float
    kd_scale: float
    joint_damping: float
    motor_strength: float
    action_latency: float
    com_offset_x: float

    def as_privileged_vector(self):
        """
        Normalise each parameter to ~[-1, 1] using its own sampling range.

        Normalisation matters: raw values span 0.3 (friction) to 3.0 (mass),
        and an unnormalised encoder input would let mass dominate the latent
        purely because of its magnitude.
        """
        out = np.empty(PRIV_DIM, dtype=np.float32)
        for i, (name, low, high, _) in enumerate(PARAM_SPECS):
            value = getattr(self, name)
            mid = 0.5 * (low + high)
            half = 0.5 * (high - low)
            out[i] = 0.0 if half == 0 else (value - mid) / half
        return out

    @classmethod
    def nominal(cls):
        """Un-randomised values — used for evaluation and for `--no-domain-rand`."""
        return cls(**{name: nominal for name, _, _, nominal in PARAM_SPECS})

    @classmethod
    def sample(cls, rng: np.random.Generator):
        return cls(**{
            name: float(rng.uniform(low, high))
            for name, low, high, _ in PARAM_SPECS
        })


def apply_to_model(model, params: DomainParams, base):
    """
    Write sampled parameters into a live MjModel.

    `base` is a dict of pristine arrays captured once at construction (see
    Go2Env._capture_base_model), because MjModel is mutated in place and the
    original values would otherwise be lost after the first episode.

    Returns the (kp, kd) multipliers, which are applied in the control loop
    rather than in the model.
    """
    # Friction: MuJoCo geom_friction is (sliding, torsional, rolling).
    model.geom_friction[:, 0] = base["geom_friction"][:, 0] * (
        params.friction / 0.60
    )

    # Trunk mass. Body 1 is the base/trunk (body 0 is always the world).
    model.body_mass[1] = max(0.1, base["body_mass"][1] + params.added_mass)

    # Centre-of-mass offset on the trunk.
    model.body_ipos[1, 0] = base["body_ipos"][1, 0] + params.com_offset_x

    # Joint damping — skip the first 6 dofs, which belong to the free base.
    model.dof_damping[6:] = base["dof_damping"][6:] * params.joint_damping

    # Per-joint torque limits stand in for motor strength.
    model.actuator_ctrlrange[:] = (
        base["actuator_ctrlrange"] * params.motor_strength
    )

    return params.kp_scale, params.kd_scale
