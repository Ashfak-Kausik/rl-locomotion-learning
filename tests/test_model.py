"""
The MJCF structural contract.

These assertions encode facts the inference pipeline depends on absolutely.
If any of them changes, the observation vector silently means something
different and the robot stops walking with no error message.
"""

import mujoco
import numpy as np
import pytest

# Actuator order as declared in scenes/go2_model/go2.xml lines 189-200.
EXPECTED_ACTUATORS = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
]
HIP_INDICES = [0, 3, 6, 9]


def test_state_dimensions(model):
    """
    19 qpos, 18 qvel, 12 actuators.

    nq != nv because the floating base is a free joint: orientation needs 4
    numbers in position space (quaternion) but only 3 in velocity space
    (angular velocity). That one-off is why joint angles start at qpos[7]
    while joint velocities start at qvel[6].
    """
    assert model.nq == 19, f"expected 19 qpos, got {model.nq}"
    assert model.nv == 18, f"expected 18 qvel, got {model.nv}"
    assert model.nu == 12, f"expected 12 actuators, got {model.nu}"
    assert model.nq - model.nv == 1, "free-joint quaternion offset broken"


def test_physics_runs_at_500hz(model):
    """The 50 Hz policy rate is timestep x DECIMATION. Both must hold."""
    from harness import DECIMATION

    assert model.opt.timestep == pytest.approx(0.002), (
        f"physics timestep is {model.opt.timestep}, expected 0.002 s"
    )
    policy_hz = 1.0 / (model.opt.timestep * DECIMATION)
    assert policy_hz == pytest.approx(50.0), (
        f"policy rate is {policy_hz} Hz; the pretrained policy requires 50 Hz"
    )


def test_actuator_order(model):
    """Joint ordering is FL, FR, RL, RR x (hip, thigh, calf)."""
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)
    ]
    assert names == EXPECTED_ACTUATORS, (
        "actuator order changed — every index-based slice in the pipeline "
        f"is now wrong.\n  expected: {EXPECTED_ACTUATORS}\n  got:      {names}"
    )


def test_hip_indices_are_hips(model):
    """
    Indices 0, 3, 6, 9 must be the four hip (abduction) joints.

    These are the indices that receive HIP_SCALE_REDUCTION. Pointing that
    mask at the wrong joints makes the robot splay its legs and collapse.
    """
    for idx in HIP_INDICES:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx)
        assert name.endswith("_hip"), (
            f"actuator index {idx} is '{name}', expected a hip joint — "
            "HIP_SCALE_REDUCTION would be applied to the wrong joint"
        )


def test_actuators_are_torque_motors(model):
    """
    The Go2 MJCF uses <motor> actuators, so data.ctrl is TORQUE in Nm, not a
    target angle. This is the entire reason the repo hand-writes a PD loop.
    If someone swaps in <position> actuators, the PD controller becomes a
    catastrophic double-controller.
    """
    for i in range(model.nu):
        assert model.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_FIXED, (
            f"actuator {i} is not a plain fixed-gain motor; data.ctrl may no "
            "longer be a torque, which breaks the PD control law"
        )
        assert model.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_NONE, (
            f"actuator {i} has a bias term — this looks like a <position> "
            "actuator, not a torque <motor>"
        )


def test_home_keyframe_exists_and_differs_from_training_pose(model):
    """
    Two poses exist and they are NOT the same:
      - MJCF 'home' keyframe:  0, 0.9, -1.8 per leg
      - training default pose: +-0.1, 0.8/1.0, -1.5

    Every runtime script resets to the keyframe and then overrides the joint
    block. This test documents *why* that override is not redundant.
    """
    from harness import DEFAULT_JOINT_POS

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    assert key_id >= 0, "the 'home' keyframe is missing from the MJCF"

    keyframe_joints = model.key_qpos[key_id][7:19]
    assert not np.allclose(keyframe_joints, DEFAULT_JOINT_POS), (
        "the MJCF keyframe now equals the training default pose. Either the "
        "model changed or DEFAULT_JOINT_POS did — verify against the "
        "walk-these-ways training config before trusting the pipeline."
    )


def test_hip_sign_convention():
    """
    Left legs (FL, RL) have POSITIVE hip defaults; right legs (FR, RR) NEGATIVE.

    The alternating signs look like a typo and have been "fixed" by mistake
    before (see 04_build_obs_vector.py, which has them inverted).
    """
    from harness import DEFAULT_JOINT_POS

    fl_hip, fr_hip, rl_hip, rr_hip = (DEFAULT_JOINT_POS[i] for i in HIP_INDICES)
    assert fl_hip > 0, f"FL hip default should be positive, got {fl_hip}"
    assert fr_hip < 0, f"FR hip default should be negative, got {fr_hip}"
    assert rl_hip > 0, f"RL hip default should be positive, got {rl_hip}"
    assert rr_hip < 0, f"RR hip default should be negative, got {rr_hip}"
    assert fl_hip == pytest.approx(-fr_hip), "hip defaults should mirror"
    assert rl_hip == pytest.approx(-rr_hip), "hip defaults should mirror"


def test_physics_is_stable_under_pd_control(model, data):
    """
    Integration smoke test: hold the default pose with the PD law for 1 s.
    The robot must not fall, explode, or produce NaNs.
    """
    from harness import DEFAULT_JOINT_POS, KP, KD

    for _ in range(500):  # 1 second at 500 Hz
        q = data.qpos[7:19]
        qd = data.qvel[6:18]
        data.ctrl[:] = KP * (DEFAULT_JOINT_POS - q) - KD * qd
        mujoco.mj_step(model, data)

    assert np.all(np.isfinite(data.qpos)), "simulation produced NaN/inf"
    assert data.qpos[2] > 0.15, (
        f"robot collapsed to {data.qpos[2]:.3f} m while merely holding a pose"
    )
