"""
Go2 locomotion training environment.

The one thing that matters: **this environment emits the exact same 70-dim
observation that Stage 2 built.** Same field order, same scale factors, same
default pose, same gait clock, same dual-rate PD control. Everything else here
is negotiable; that is not.

The payoff is that a policy trained here can be evaluated by
`stage2-go2-mujoco-inference/experiments/harness.py` with zero modifications —
same scenes, same metrics, same figures — and compared directly against the
walk-these-ways baseline numbers in EXPERIMENT_FINDINGS.md.

To guarantee it rather than hope for it, the constants are IMPORTED from
harness.py rather than copied. If harness.py changes, this environment changes
with it, and tests/test_stage3_contract.py fails loudly if they ever diverge.

Backend note
------------
This runs on plain MuJoCo (CPU), so it works on any machine and is what the
test suite exercises. It is not fast: expect ~2-4k steps/s single-threaded, so
a serious training run wants either many parallel workers or the MJX/GPU path
described in the Stage 3 README. The environment interface is deliberately
backend-agnostic so that swap is contained.
"""

import os
import sys
from collections import deque

import mujoco
import numpy as np

# Import the contract from Stage 2 — never re-declare it.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
_STAGE2 = os.path.join(_REPO_ROOT, "stage2-go2-mujoco-inference")
for _p in (_STAGE2, os.path.join(_STAGE2, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from harness import (  # noqa: E402
    ACTION_SCALE,
    COMMANDS_SCALE,
    DECIMATION,
    DEFAULT_JOINT_POS,
    GAIT_PRESETS,
    HIP_SCALE_REDUCTION,
    HISTORY_LEN,
    KD,
    KP,
    OBS_DIM,
    OBS_SCALES,
    STEP_FREQUENCY,
    FALL_HEIGHT_THRESHOLD,
    FALL_TILT_THRESHOLD_DEG,
    build_obs,
    quat_rotate_inverse,
    tilt_angle_deg,
)

from . import rewards as R  # noqa: E402
from .domain_rand import DomainParams, apply_to_model  # noqa: E402

SCENES_DIR = os.path.join(_STAGE2, "scenes")
HIP_INDICES = (0, 3, 6, 9)


class Go2Env:
    """
    Single-instance MuJoCo locomotion environment.

    Deliberately not a `gymnasium.Env` subclass: the observation the policy
    consumes is the flattened 30-frame HISTORY (2100 dims), not a single
    frame, and the privileged vector is returned alongside it for RMA phase 1.
    That does not fit the Gym signature cleanly, and pretending otherwise would
    obscure the contract. `reset()`/`step()` keep Gym-like semantics.
    """

    def __init__(self, cfg, seed=0, scene="go2_flat.xml"):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.scene_name = None
        self._model_cache = {}
        self._load_scene(scene)

        self.action_dim = 12
        self.obs_dim = OBS_DIM
        self.history_dim = HISTORY_LEN * OBS_DIM

        self.dt_policy = self.model.opt.timestep * DECIMATION  # 0.02 s (50 Hz)
        self.max_episode_steps = int(cfg.episode_seconds / self.dt_policy)

        # Per-joint action scale: hips get the extra 0.5x reduction.
        self.action_scale = np.full(12, ACTION_SCALE)
        for hip in HIP_INDICES:
            self.action_scale[hip] *= HIP_SCALE_REDUCTION

        self._max_return = None
        self.reset()

    # ------------------------------------------------------------------
    # Model handling
    # ------------------------------------------------------------------
    def _load_scene(self, scene_name):
        """Load (and cache) a scene. MuJoCo needs an ABSOLUTE path here."""
        if scene_name == self.scene_name:
            return
        if scene_name not in self._model_cache:
            path = os.path.abspath(os.path.join(SCENES_DIR, scene_name))
            if not os.path.isfile(path):
                raise FileNotFoundError(f"scene not found: {path}")
            self._model_cache[scene_name] = mujoco.MjModel.from_xml_path(path)

        self.model = self._model_cache[scene_name]
        self.data = mujoco.MjData(self.model)
        self.scene_name = scene_name
        self._base = self._capture_base_model(self.model)

    @staticmethod
    def _capture_base_model(model):
        """
        Snapshot the pristine model parameters.

        MjModel is mutated in place by domain randomisation, so without this
        the original values are lost after the first episode and randomisation
        compounds across episodes.
        """
        return {
            "geom_friction": model.geom_friction.copy(),
            "body_mass": model.body_mass.copy(),
            "body_ipos": model.body_ipos.copy(),
            "dof_damping": model.dof_damping.copy(),
            "actuator_ctrlrange": model.actuator_ctrlrange.copy(),
        }

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------
    def reset(self, scene=None, command=None, domain_params=None):
        if scene is not None:
            self._load_scene(scene)

        # Same reset procedure as Stage 2: keyframe, then override with the
        # training default pose. Skipping the override offsets every joint obs.
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[7:19] = DEFAULT_JOINT_POS
        self.data.qpos[2] = 0.30

        # Initial-condition randomisation, matching harness.run_trial's
        # magnitudes so training and evaluation see the same start distribution.
        self.data.qpos[7:19] += self.rng.uniform(-0.02, 0.02, size=12)
        self.data.qpos[2] += self.rng.uniform(-0.01, 0.01)
        yaw = self.rng.uniform(-0.05, 0.05)
        self.data.qpos[3] = np.cos(yaw / 2.0)
        self.data.qpos[4:6] = 0.0
        self.data.qpos[6] = np.sin(yaw / 2.0)
        self.data.qvel[6:18] += self.rng.uniform(-0.05, 0.05, size=12)

        # Domain randomisation for this episode.
        if domain_params is None:
            domain_params = (
                DomainParams.sample(self.rng) if self.cfg.domain_rand
                else DomainParams.nominal()
            )
        self.domain_params = domain_params
        self.kp_scale, self.kd_scale = apply_to_model(
            self.model, domain_params, self._base)
        self.privileged = domain_params.as_privileged_vector()
        self._latency_steps = int(round(domain_params.action_latency))

        mujoco.mj_forward(self.model, self.data)

        # Command for this episode.
        if command is None:
            command = (self.rng.uniform(0.0, 0.5), 0.0, 0.0)
        self.cmd_vx, self.cmd_vy, self.cmd_yaw = command
        if self.cfg.randomize_gait:
            self.gait = self.rng.choice(self.cfg.gait_pool)
        else:
            self.gait = self.cfg.gait
        self.gait_params = GAIT_PRESETS[self.gait]

        # Rolling state.
        self.obs_history = deque(
            [np.zeros(OBS_DIM, dtype=np.float32) for _ in range(HISTORY_LEN)],
            maxlen=HISTORY_LEN,
        )
        self.last_action = np.zeros(12)
        self.prev_action = np.zeros(12)
        self._action_queue = deque(
            [np.zeros(12)] * (self._latency_steps + 1),
            maxlen=self._latency_steps + 1,
        )
        self.joint_targets = DEFAULT_JOINT_POS.copy()
        self.prev_joint_vel = np.zeros(12)
        self.gait_phase_t = 0.0
        self.step_count = 0
        self.episode_return = 0.0
        self.reward_breakdown = {}

        self._push_obs()
        return self.get_observation(), self.privileged.copy()

    def _command_vector(self):
        """The 15-dim command block, ordered exactly as Stage 2 expects."""
        return np.array([
            self.cmd_vx, self.cmd_vy, self.cmd_yaw, 0.0, STEP_FREQUENCY,
            *self.gait_params, 0.5, 0.06, 0.0, 0.0, 0.0, 0.0, 0.0,
        ])

    def _push_obs(self):
        """Build one observation frame with Stage 2's own build_obs()."""
        obs = build_obs(
            self.data,
            self._command_vector(),
            self.gait_params,
            self.prev_action,
            self.last_action,
            self.gait_phase_t,
        )
        assert obs.shape == (OBS_DIM,), f"contract violated: {obs.shape}"
        self.obs_history.append(obs.astype(np.float32))

    def get_observation(self):
        """Flattened 30-frame history — (2100,), the network's actual input."""
        return np.concatenate(self.obs_history).astype(np.float32)

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------
    def step(self, action):
        """
        Advance one POLICY step (= DECIMATION physics steps).

        Returns (obs, privileged, reward, terminated, truncated, info) —
        Gym-style, with the privileged vector added for RMA phase 1.
        """
        action = np.clip(np.asarray(action, dtype=np.float64), -100.0, 100.0)

        # Action latency: what the motors receive may be a few steps stale.
        self._action_queue.append(action)
        applied = self._action_queue[0]

        self.prev_action = self.last_action.copy()
        self.last_action = action.copy()
        self.joint_targets = DEFAULT_JOINT_POS + applied * self.action_scale

        # Advance the gait clock on the POLICY clock, not the physics clock.
        self.gait_phase_t = (
            self.gait_phase_t + STEP_FREQUENCY * self.dt_policy) % 1.0

        # Inner loop: PD torque control at 500 Hz.
        kp = KP * self.kp_scale
        kd = KD * self.kd_scale
        torques = np.zeros(12)
        for _ in range(DECIMATION):
            q = self.data.qpos[7:19]
            qd = self.data.qvel[6:18]
            torques = kp * (self.joint_targets - q) - kd * qd
            self.data.ctrl[:] = torques
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        self._push_obs()

        reward, terminated = self._compute_reward(torques)
        self.episode_return += reward
        truncated = self.step_count >= self.max_episode_steps

        info = {
            "height": float(self.data.qpos[2]),
            "vx": float(self.data.qvel[0]),
            "vy": float(self.data.qvel[1]),
            "cmd_vx": self.cmd_vx,
            "episode_return": self.episode_return,
            "reward_terms": self.reward_breakdown,
            "scene": self.scene_name,
        }
        if terminated or truncated:
            info["episode"] = {
                "r": self.episode_return,
                "l": self.step_count,
                "max_r": self.max_possible_return(),
            }

        return (self.get_observation(), self.privileged.copy(),
                reward, terminated, truncated, info)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _compute_reward(self, torques):
        d = self.data
        quat = d.qpos[3:7]
        proj_grav = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        joint_vel = d.qvel[6:18].copy()

        # Terminate on the SAME criteria harness.py uses, so "survived" means
        # the same thing in training and in evaluation.
        height = float(d.qpos[2])
        tilt = tilt_angle_deg(quat)
        fell = height < FALL_HEIGHT_THRESHOLD or tilt > FALL_TILT_THRESHOLD_DEG

        terms = {
            "tracking_lin_vel": R.tracking_lin_vel(
                self.cmd_vx, self.cmd_vy, d.qvel[0], d.qvel[1],
                self.cfg.tracking_sigma),
            "tracking_ang_vel": R.tracking_ang_vel(
                self.cmd_yaw, d.qvel[5], self.cfg.tracking_sigma),
            "lateral_drift": R.lateral_drift(d.qvel[1]),
            "vertical_velocity": R.vertical_velocity(d.qvel[2]),
            "body_orientation": R.body_orientation(proj_grav),
            "body_height": R.body_height(height),
            "joint_torque": R.joint_torque(torques),
            "joint_velocity": R.joint_velocity(joint_vel),
            "joint_acceleration": R.joint_acceleration(
                joint_vel, self.prev_joint_vel, self.dt_policy),
            "action_rate": R.action_rate(self.last_action, self.prev_action),
            "action_magnitude": R.action_magnitude(self.last_action),
            "alive": R.alive(),
        }
        self.prev_joint_vel = joint_vel

        total, breakdown = R.compute_total(terms, self.cfg.reward_weights)
        self.reward_breakdown = breakdown

        if fell:
            total += self.cfg.fall_penalty

        return total, fell

    def max_possible_return(self):
        """
        Upper bound on episode return: perfect tracking, zero penalties, no
        early termination. Used to normalise curriculum progress so the
        promotion threshold means the same thing at every level.
        """
        if self._max_return is None:
            w = self.cfg.reward_weights
            per_step = (w.get("tracking_lin_vel", 0.0)
                        + w.get("tracking_ang_vel", 0.0)
                        + w.get("alive", 0.0))
            self._max_return = per_step * self.max_episode_steps
        return self._max_return
