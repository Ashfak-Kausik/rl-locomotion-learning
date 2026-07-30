"""
Shared pytest fixtures.

Design rule for this whole suite: **no test may require the walk-these-ways
policy checkpoints.** They are a large external artefact most contributors will
not have (see docs/DEPENDENCIES.md §6). Everything here exercises the
contracts, the model, the scenes and the constants — all of which are in-repo.

Tests that genuinely need weights must be marked `@pytest.mark.needs_policy`
and will skip automatically when the files are absent.
"""

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE2 = REPO_ROOT / "stage2-go2-mujoco-inference"
SCENES = STAGE2 / "scenes"
EXPERIMENTS = STAGE2 / "experiments"

# Make `paths` and `harness` importable.
sys.path.insert(0, str(STAGE2))
sys.path.insert(0, str(EXPERIMENTS))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "needs_policy: requires the walk-these-ways .jit checkpoints"
    )
    config.addinivalue_line(
        "markers", "slow: takes more than a couple of seconds"
    )


def pytest_collection_modifyitems(config, items):
    """Skip policy-dependent tests when the checkpoints are not present."""
    import paths

    if paths.policy_available():
        return
    skip = pytest.mark.skip(
        reason=f"policy checkpoints not found in {paths.POLICY_DIR} "
               "(see docs/DEPENDENCIES.md §6)"
    )
    for item in items:
        if "needs_policy" in item.keywords:
            item.add_marker(skip)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT


@pytest.fixture(scope="session")
def scenes_dir():
    return SCENES


@pytest.fixture(scope="session")
def flat_scene():
    """Absolute path to the flat scene. MuJoCo needs absolute paths here."""
    return str((SCENES / "go2_flat.xml").resolve())


# ---------------------------------------------------------------------------
# MuJoCo model
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def model(flat_scene):
    """Loaded MjModel for the flat scene. Session-scoped: parsing is slow."""
    import mujoco
    return mujoco.MjModel.from_xml_path(flat_scene)


@pytest.fixture
def data(model):
    """Fresh MjData per test, reset to the training default pose."""
    import mujoco
    import numpy as np
    from harness import DEFAULT_JOINT_POS

    d = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, d, key_id)
    d.qpos[7:19] = DEFAULT_JOINT_POS
    d.qpos[2] = 0.30
    mujoco.mj_forward(model, d)
    return d


# ---------------------------------------------------------------------------
# Static source analysis
# ---------------------------------------------------------------------------
# The numbered scripts execute on import (06 launches a viewer, 01 opens a
# window). They can never be imported by a test. Parsing them with `ast` is the
# only safe way to inspect their module-level constants — which is exactly what
# the cross-file drift guard in test_constants.py needs.
@pytest.fixture(scope="session")
def module_constants():
    """
    Map: filename -> {CONSTANT_NAME: literal_value}

    Only literal assignments are captured (numbers, strings, lists, dicts).
    Anything computed at runtime is skipped rather than guessed at.
    """
    targets = [
        STAGE2 / "06_run_policy.py",
        STAGE2 / "07_run_policy_interactive.py",
        STAGE2 / "08_view_scene.py",
        EXPERIMENTS / "harness.py",
    ]
    def literal(node):
        """Evaluate a literal, unwrapping a single-arg call like np.array([…])."""
        try:
            return ast.literal_eval(node)
        except (ValueError, TypeError, SyntaxError):
            if isinstance(node, ast.Call) and node.args:
                try:
                    return ast.literal_eval(node.args[0])
                except (ValueError, TypeError, SyntaxError):
                    pass
        raise ValueError("not a literal")

    def record(consts, target, value_node):
        if isinstance(target, ast.Name) and target.id.isupper():
            try:
                consts[target.id] = literal(value_node)
            except ValueError:
                pass
        # Tuple unpacking: `KP, KD = 25.0, 0.6` — used in harness/07/08.
        elif isinstance(target, ast.Tuple) and isinstance(value_node, ast.Tuple):
            for t, v in zip(target.elts, value_node.elts):
                record(consts, t, v)

    out = {}
    for path in targets:
        tree = ast.parse(path.read_text(), filename=str(path))
        consts = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    record(consts, target, node.value)
        out[path.name] = consts
    return out
