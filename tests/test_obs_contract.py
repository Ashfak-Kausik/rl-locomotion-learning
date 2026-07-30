"""
The 70-dimensional observation contract.

This is the most important test file in the repository.

The observation layout is fixed by a pretrained policy binary. It is documented
nowhere upstream — it was recovered by reading the walk-these-ways training
source. A wrong field order, a wrong scale factor or a flipped sign does not
raise: the robot just walks worse, or falls over, with no error.

These tests are the only automated defence of that contract.

Layout (see docs/ARCHITECTURE.md section 7):
    [ 0: 3]  projected gravity, body frame
    [ 3:18]  15 commands x COMMANDS_SCALE
    [18:30]  (joint_pos - DEFAULT_JOINT_POS) x 1.0
    [30:42]  joint_vel x 0.05
    [42:54]  last_action  (t-1)
    [54:66]  prev_action  (t-2)
    [66:70]  4 per-foot clock signals
"""

import numpy as np
import pytest

from harness import (
    DEFAULT_JOINT_POS,
    GAIT_PRESETS,
    OBS_SCALES,
    COMMANDS_SCALE,
    ACTION_SCALE,
    HIP_SCALE_REDUCTION,
    HISTORY_LEN,
    OBS_DIM,
    build_obs,
    quat_rotate_inverse,
    tilt_angle_deg,
)

# Field boundaries. These are the contract.
SLICE_GRAVITY = slice(0, 3)
SLICE_COMMANDS = slice(3, 18)
SLICE_JOINT_POS = slice(18, 30)
SLICE_JOINT_VEL = slice(30, 42)
SLICE_LAST_ACTION = slice(42, 54)
SLICE_PREV_ACTION = slice(54, 66)
SLICE_CLOCK = slice(66, 70)

IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # (w, x, y, z)


def _commands(vx=0.5, vy=0.0, yaw=0.0, gait="trot"):
    """Build the 15-dim command vector exactly as harness.run_trial does."""
    from harness import STEP_FREQUENCY
    return np.array([
        vx, vy, yaw, 0.0, STEP_FREQUENCY,
        *GAIT_PRESETS[gait], 0.5, 0.06, 0.0, 0.0, 0.0, 0.0, 0.0,
    ])


# ===========================================================================
# Dimensions
# ===========================================================================
def test_obs_dim_is_70():
    assert OBS_DIM == 70


def test_history_dims_match_network_input():
    """30 frames x 70 dims = 2100; body net takes 2100 + 2 latent = 2102."""
    assert HISTORY_LEN == 30
    assert HISTORY_LEN * OBS_DIM == 2100
    assert HISTORY_LEN * OBS_DIM + 2 == 2102


def test_build_obs_returns_exactly_70_dims(data):
    obs = build_obs(data, _commands(), GAIT_PRESETS["trot"],
                    np.zeros(12), np.zeros(12), 0.0)
    assert obs.shape == (70,), f"got {obs.shape}, contract requires (70,)"
    assert np.all(np.isfinite(obs)), "observation contains NaN/inf"


def test_field_widths_sum_to_70():
    """The concatenation must partition [0, 70) with no gaps or overlaps."""
    slices = [SLICE_GRAVITY, SLICE_COMMANDS, SLICE_JOINT_POS, SLICE_JOINT_VEL,
              SLICE_LAST_ACTION, SLICE_PREV_ACTION, SLICE_CLOCK]
    widths = [s.stop - s.start for s in slices]
    assert widths == [3, 15, 12, 12, 12, 12, 4]
    assert sum(widths) == 70
    for a, b in zip(slices, slices[1:]):
        assert a.stop == b.start, "field boundaries are not contiguous"


# ===========================================================================
# Field placement — each test pins one block to its index range
# ===========================================================================
def test_gravity_block_is_upright_at_rest(data):
    """Standing upright, projected gravity must be approximately [0, 0, -1]."""
    obs = build_obs(data, _commands(), GAIT_PRESETS["trot"],
                    np.zeros(12), np.zeros(12), 0.0)
    grav = obs[SLICE_GRAVITY]
    np.testing.assert_allclose(grav, [0.0, 0.0, -1.0], atol=1e-6)


def test_commands_block_is_scaled(data):
    """Commands appear at [3:18], each multiplied by its own scale factor."""
    cmd = _commands(vx=1.0, vy=0.5, yaw=0.25)
    obs = build_obs(data, cmd, GAIT_PRESETS["trot"],
                    np.zeros(12), np.zeros(12), 0.0)
    np.testing.assert_allclose(obs[SLICE_COMMANDS], cmd * COMMANDS_SCALE)

    # Spot-check the three that matter most, against the training config.
    assert obs[3] == pytest.approx(1.0 * 2.0)    # lin_vel_x x 2.0
    assert obs[4] == pytest.approx(0.5 * 2.0)    # lin_vel_y x 2.0
    assert obs[5] == pytest.approx(0.25 * 0.25)  # ang_vel_yaw x 0.25


def test_joint_positions_are_relative_to_default_pose(data):
    """
    At the training default pose, the joint-position block must be ~zero.
    A non-zero block here means the keyframe override was skipped, which
    offsets every joint observation by a constant.
    """
    obs = build_obs(data, _commands(), GAIT_PRESETS["trot"],
                    np.zeros(12), np.zeros(12), 0.0)
    np.testing.assert_allclose(obs[SLICE_JOINT_POS], np.zeros(12), atol=1e-9)


def test_joint_position_offset_is_reported(data):
    """Perturb one joint; only that element of the block may change."""
    data.qpos[7 + 4] += 0.1          # FR_thigh, i.e. joint index 4
    obs = build_obs(data, _commands(), GAIT_PRESETS["trot"],
                    np.zeros(12), np.zeros(12), 0.0)
    block = obs[SLICE_JOINT_POS]
    assert block[4] == pytest.approx(0.1 * OBS_SCALES["dof_pos"])
    others = np.delete(block, 4)
    np.testing.assert_allclose(others, np.zeros(11), atol=1e-9)


def test_joint_velocities_are_scaled_by_0_05(data):
    """dof_vel scale is 0.05 — velocities reach +-20 rad/s and must be tamed."""
    data.qvel[6:18] = 2.0
    obs = build_obs(data, _commands(), GAIT_PRESETS["trot"],
                    np.zeros(12), np.zeros(12), 0.0)
    np.testing.assert_allclose(obs[SLICE_JOINT_VEL], np.full(12, 2.0 * 0.05))
    assert OBS_SCALES["dof_vel"] == 0.05


def test_action_history_order_is_last_then_prev(data):
    """
    [42:54] is the MOST RECENT action, [54:66] the one before it.
    Swapping these gives the policy a time-reversed view of its own output.
    """
    last = np.full(12, 0.11)
    prev = np.full(12, 0.22)
    obs = build_obs(data, _commands(), GAIT_PRESETS["trot"],
                    prev, last, 0.0)
    np.testing.assert_allclose(obs[SLICE_LAST_ACTION], last)
    np.testing.assert_allclose(obs[SLICE_PREV_ACTION], prev)


def test_clock_is_the_final_four_dims(data):
    """
    Clock signals live at [66:70], NOT at [42:46].
    The superseded 04_build_obs_vector.py places them at 42 — that file is a
    known-wrong historical artefact. This test pins the correct position.
    """
    obs = build_obs(data, _commands(), GAIT_PRESETS["trot"],
                    np.zeros(12), np.zeros(12), 0.25)
    clock = obs[SLICE_CLOCK]
    assert clock.shape == (4,)
    assert np.all(np.abs(clock) <= 1.0), "clock signals must be sin() outputs"


# ===========================================================================
# Gait conditioning
# ===========================================================================
def test_gait_presets_are_the_expected_triples():
    """Each gait is a (phase, offset, bound) triple. Same net, 3 numbers."""
    assert GAIT_PRESETS["trot"] == (0.5, 0.0, 0.0)
    assert GAIT_PRESETS["pace"] == (0.0, 0.5, 0.0)
    assert GAIT_PRESETS["bound"] == (0.0, 0.0, 0.5)


def test_gaits_produce_distinct_clock_signals(data):
    """
    If two gaits produced identical clocks, gait conditioning would be a no-op
    and Experiment 2's results would be meaningless.
    """
    clocks = {}
    for gait in ("trot", "pace", "bound"):
        obs = build_obs(data, _commands(gait=gait), GAIT_PRESETS[gait],
                        np.zeros(12), np.zeros(12), 0.3)
        clocks[gait] = obs[SLICE_CLOCK]

    for a, b in (("trot", "pace"), ("trot", "bound"), ("pace", "bound")):
        assert not np.allclose(clocks[a], clocks[b]), (
            f"{a} and {b} produce identical clock signals — "
            "gait conditioning is broken"
        )


def test_trot_moves_feet_in_diagonal_pairs(data):
    """
    Trot = diagonal pairs in antiphase. With phase=0.5, feet 0 and 3 share a
    phase and feet 1 and 2 share a phase, half a cycle apart.
    """
    obs = build_obs(data, _commands(gait="trot"), GAIT_PRESETS["trot"],
                    np.zeros(12), np.zeros(12), 0.1)
    c = obs[SLICE_CLOCK]
    assert c[0] == pytest.approx(c[3], abs=1e-9), "trot pair 0/3 out of sync"
    assert c[1] == pytest.approx(c[2], abs=1e-9), "trot pair 1/2 out of sync"
    assert not np.isclose(c[0], c[1]), "trot pairs should be in antiphase"


def test_clock_wraps_over_a_full_cycle(data):
    """Phase is modulo 1.0, so t and t+1 must give identical clocks."""
    kw = dict(commands_vec=_commands(), gait_params=GAIT_PRESETS["trot"],
              prev_action=np.zeros(12), last_action=np.zeros(12))
    a = build_obs(data, gait_phase_t=0.37, **kw)[SLICE_CLOCK]
    b = build_obs(data, gait_phase_t=1.37, **kw)[SLICE_CLOCK]
    np.testing.assert_allclose(a, b, atol=1e-9)


# ===========================================================================
# Maths helpers
# ===========================================================================
def test_quat_rotate_inverse_identity_is_noop():
    v = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(quat_rotate_inverse(IDENTITY_QUAT, v), v,
                               atol=1e-12)


def test_quat_rotate_inverse_preserves_length():
    """Rotation is an isometry — magnitude must not change."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        v = rng.normal(size=3)
        out = quat_rotate_inverse(q, v)
        assert np.linalg.norm(out) == pytest.approx(np.linalg.norm(v), rel=1e-9)


def test_quat_rotate_inverse_180_degree_yaw():
    """A half-turn about z flips x and y, leaves z alone."""
    q = np.array([0.0, 0.0, 0.0, 1.0])  # w=0, z=1 -> 180 deg about z
    out = quat_rotate_inverse(q, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(out, [-1.0, -2.0, 3.0], atol=1e-12)


def test_tilt_angle_upright_is_zero():
    assert tilt_angle_deg(IDENTITY_QUAT) == pytest.approx(0.0, abs=1e-6)


def test_tilt_angle_90_degrees_on_its_side():
    """A quarter-turn about x puts the body on its side: 90 deg of tilt."""
    s = np.sqrt(0.5)
    q = np.array([s, s, 0.0, 0.0])
    assert tilt_angle_deg(q) == pytest.approx(90.0, abs=1e-6)


def test_tilt_angle_is_never_nan_for_unit_quaternions():
    """arccos must stay clamped — floating-point drift once produced NaN."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        assert np.isfinite(tilt_angle_deg(q))


# ===========================================================================
# Action scaling
# ===========================================================================
def test_hip_action_scale_is_halved():
    """
    Hips (indices 0, 3, 6, 9) get ACTION_SCALE x HIP_SCALE_REDUCTION;
    thighs and calves get the full ACTION_SCALE.
    """
    scale = np.full(12, ACTION_SCALE)
    for hip in (0, 3, 6, 9):
        scale[hip] *= HIP_SCALE_REDUCTION

    assert ACTION_SCALE == 0.25
    assert HIP_SCALE_REDUCTION == 0.5
    for hip in (0, 3, 6, 9):
        assert scale[hip] == pytest.approx(0.125)
    for other in (1, 2, 4, 5, 7, 8, 10, 11):
        assert scale[other] == pytest.approx(0.25)


def test_zero_action_maps_to_the_default_pose():
    """An all-zero action must be exactly the default pose, on every joint."""
    scale = np.full(12, ACTION_SCALE)
    for hip in (0, 3, 6, 9):
        scale[hip] *= HIP_SCALE_REDUCTION
    targets = DEFAULT_JOINT_POS + np.zeros(12) * scale
    np.testing.assert_allclose(targets, DEFAULT_JOINT_POS)
