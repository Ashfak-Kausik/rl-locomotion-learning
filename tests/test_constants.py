"""
Cross-file constant drift guard.

DEFAULT_JOINT_POS, OBS_SCALES, the PD gains and the timing constants are
copy-pasted across 06_run_policy.py, 07_run_policy_interactive.py,
08_view_scene.py and experiments/harness.py.

That duplication is deliberate — the numbered scripts are a curriculum meant to
be read standalone (see docs/REVERSE-ENGINEERING.md section 5). But deliberate
duplication still drifts: a fix applied to harness.py will not reach 07.

These tests make drift a test failure instead of a silent behavioural
difference between "the demo" and "the experiments".

The scripts execute on import (06 launches a viewer), so their constants are
extracted by parsing the source with `ast` — see the `module_constants`
fixture in conftest.py.
"""

import numpy as np
import pytest

# Constants that MUST agree everywhere they appear.
SHARED_SCALARS = [
    "ACTION_SCALE",
    "HIP_SCALE_REDUCTION",
    "KP",
    "KD",
    "DECIMATION",
    "HISTORY_LEN",
    "OBS_DIM",
]

FILES = [
    "06_run_policy.py",
    "07_run_policy_interactive.py",
    "08_view_scene.py",
    "harness.py",
]

# harness.py is the reference implementation (docs/REVERSE-ENGINEERING.md).
REFERENCE = "harness.py"


@pytest.mark.parametrize("const", SHARED_SCALARS)
def test_scalar_constants_agree_across_files(module_constants, const):
    values = {
        fname: consts[const]
        for fname, consts in module_constants.items()
        if const in consts
    }
    assert len(values) >= 2, (
        f"{const} was found in fewer than 2 files — the parser may have "
        f"broken. Found in: {list(values)}"
    )
    distinct = set(values.values())
    assert len(distinct) == 1, (
        f"{const} has DRIFTED between files: {values}\n"
        f"harness.py is the reference implementation."
    )


def test_default_joint_pos_agrees_across_files(module_constants):
    """
    The hip signs in particular. 04_build_obs_vector.py has them inverted
    (a known historical bug); none of these four files may repeat it.
    """
    values = {
        fname: np.asarray(consts["DEFAULT_JOINT_POS"], dtype=float)
        for fname, consts in module_constants.items()
        if "DEFAULT_JOINT_POS" in consts
    }
    assert len(values) == len(FILES), (
        f"expected DEFAULT_JOINT_POS in all {len(FILES)} files, "
        f"found in {list(values)}"
    )

    reference = values[REFERENCE]
    assert reference.shape == (12,)
    for fname, arr in values.items():
        np.testing.assert_allclose(
            arr, reference,
            err_msg=f"DEFAULT_JOINT_POS in {fname} differs from {REFERENCE}",
        )


def test_obs_scales_agree_across_files(module_constants):
    """A drifted scale factor silently feeds the policy mis-normalised input."""
    values = {
        fname: consts["OBS_SCALES"]
        for fname, consts in module_constants.items()
        if "OBS_SCALES" in consts
    }
    assert len(values) == len(FILES), (
        f"expected OBS_SCALES in all {len(FILES)} files, found in {list(values)}"
    )

    reference = values[REFERENCE]
    for fname, scales in values.items():
        assert scales == reference, (
            f"OBS_SCALES in {fname} differs from {REFERENCE}:\n"
            f"  only in {fname}: "
            f"{ {k: v for k, v in scales.items() if reference.get(k) != v} }"
        )


def test_gait_presets_agree_across_files(module_constants):
    values = {
        fname: consts["GAIT_PRESETS"]
        for fname, consts in module_constants.items()
        if "GAIT_PRESETS" in consts
    }
    assert len(values) >= 3, f"GAIT_PRESETS found only in {list(values)}"

    reference = values[REFERENCE]
    for fname, presets in values.items():
        assert presets == reference, (
            f"GAIT_PRESETS in {fname} differs from {REFERENCE}: {presets}"
        )


def test_runtime_values_match_parsed_source(module_constants):
    """
    Sanity-check the ast parser itself: the values it extracted from
    harness.py must equal the values Python actually imports from it.
    """
    import harness

    parsed = module_constants[REFERENCE]
    for const in SHARED_SCALARS:
        assert parsed[const] == getattr(harness, const), (
            f"parser disagrees with the imported module on {const}"
        )
    np.testing.assert_allclose(
        np.asarray(parsed["DEFAULT_JOINT_POS"], dtype=float),
        harness.DEFAULT_JOINT_POS,
    )


def test_superseded_script_is_not_the_reference():
    """
    04_build_obs_vector.py is a known-wrong historical artefact: different
    field order AND inverted hip signs. This test asserts it still disagrees
    with harness.py, so nobody "fixes" the reference to match the broken file
    by accident.
    """
    import ast
    from conftest import STAGE2
    import harness

    tree = ast.parse((STAGE2 / "04_build_obs_vector.py").read_text())
    old = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "DEFAULT_JOINT_POS":
                    old = np.asarray(
                        ast.literal_eval(node.value.args[0]), dtype=float)
    assert old is not None, "could not parse 04_build_obs_vector.py"

    assert not np.allclose(old, harness.DEFAULT_JOINT_POS), (
        "04_build_obs_vector.py now agrees with harness.py. Either it was "
        "correctly fixed (then update this test and the docs) or harness.py "
        "was broken to match it (then revert immediately)."
    )
    # Specifically: its hip signs are the mirror of the correct ones.
    np.testing.assert_allclose(
        old[[0, 3, 6, 9]], -harness.DEFAULT_JOINT_POS[[0, 3, 6, 9]],
        err_msg="the historical hip-sign inversion is no longer what it was",
    )


def test_fall_thresholds_are_documented_values():
    """
    0.15 m is half the ~0.30 m nominal standing height; 60 deg is past the
    point of quadruped self-recovery. Both are cited in the paper.
    """
    import harness

    assert harness.FALL_HEIGHT_THRESHOLD == 0.15
    assert harness.FALL_TILT_THRESHOLD_DEG == 60.0


def test_step_frequency_matches_experiments():
    """All three experiments ran at 2.0 Hz. Changing this invalidates them."""
    import harness

    assert harness.STEP_FREQUENCY == 2.0
