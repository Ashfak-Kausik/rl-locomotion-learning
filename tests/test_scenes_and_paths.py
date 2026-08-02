"""
Scene reproducibility and path resolution.

Two guarantees are defended here:

  1. Terrain scenes regenerate byte-identically. The committed XMLs are the
     exact geometry behind the published Experiment 3 results; if the
     generator drifts, past results silently stop being reproducible.

  2. Paths resolve without machine-specific absolute paths. This is what makes
     the repo runnable on a fresh clone and inside a container.
"""

import os
import subprocess
import sys
from pathlib import Path

import mujoco
import pytest

SLOPE_ANGLES = [5, 10, 15, 20, 25]
STEP_HEIGHTS_CM = [2, 5, 8, 12, 16]

EXPECTED_SCENES = (
    ["go2_flat.xml"]
    + [f"go2_slope_{a:02d}.xml" for a in SLOPE_ANGLES]
    + [f"go2_stairs_{h:02d}.xml" for h in STEP_HEIGHTS_CM]
)


# ===========================================================================
# Scenes
# ===========================================================================
def test_all_expected_scenes_exist(scenes_dir):
    missing = [s for s in EXPECTED_SCENES if not (scenes_dir / s).is_file()]
    assert not missing, f"missing scene files: {missing}"


def test_vendored_model_is_complete(scenes_dir):
    """
    The Go2 MJCF is vendored in-tree so no external mujoco_menagerie checkout
    is needed. Every mesh it references must be present.
    """
    model_dir = scenes_dir / "go2_model"
    go2_xml = model_dir / "go2.xml"
    assert go2_xml.is_file(), "vendored go2.xml is missing"

    import re
    referenced = set(re.findall(r'file="([^"]*\.obj)"', go2_xml.read_text()))
    assert referenced, "go2.xml references no meshes — parser broken?"

    missing = [m for m in referenced if not (model_dir / m).is_file()]
    assert not missing, (
        f"go2.xml references {len(missing)} mesh(es) that are not in the "
        f"repo: {sorted(missing)}"
    )


@pytest.mark.parametrize("scene", EXPECTED_SCENES)
def test_every_scene_loads(scenes_dir, scene):
    """Each scene must parse and contain the full robot."""
    path = str((scenes_dir / scene).resolve())
    model = mujoco.MjModel.from_xml_path(path)
    assert model.nq == 19, f"{scene}: robot not included correctly"
    assert model.nu == 12, f"{scene}: actuators missing"


def test_terrain_scenes_regenerate_byte_identically(repo_root, scenes_dir):
    """
    Re-run the generator and assert git sees no change.

    This is the reproducibility guarantee for Experiment 3: the committed
    terrain is exactly what the generator produces today.
    """
    generator = (repo_root / "stage2-go2-mujoco-inference" / "experiments"
                 / "generate_terrain_scenes.py")

    before = {
        s: (scenes_dir / s).read_bytes()
        for s in EXPECTED_SCENES if s != "go2_flat.xml"  # flat is hand-written
    }

    result = subprocess.run(
        [sys.executable, str(generator)],
        capture_output=True, text=True, cwd=str(repo_root),
    )
    assert result.returncode == 0, (
        f"generator failed:\n{result.stdout}\n{result.stderr}"
    )

    changed = [
        s for s, content in before.items()
        if (scenes_dir / s).read_bytes() != content
    ]
    assert not changed, (
        f"regenerating terrain changed {len(changed)} committed scene(s): "
        f"{changed}. Past experiment results are no longer reproducible from "
        "the committed generator."
    )


def test_stairs_scene_has_the_expected_step_geometry(scenes_dir):
    """
    10 steps, 30 cm tread. Experiment 3's stall finding (a deterministic
    0.47 m against a 2.5 m run-up) only makes sense with this layout.
    """
    text = (scenes_dir / "go2_stairs_08.xml").read_text()
    assert text.count('name="step_') == 10, "expected exactly 10 steps"
    assert 'name="runup"' in text, "the flat run-up is missing"


def test_slope_scenes_are_ordered_by_angle(scenes_dir):
    """Steeper slope must mean a larger ramp inclination."""
    import re
    eulers = {}
    for angle in SLOPE_ANGLES:
        text = (scenes_dir / f"go2_slope_{angle:02d}.xml").read_text()
        m = re.search(r'name="ramp".*?euler="0 (-?[\d.]+) 0"', text, re.S)
        assert m, f"no ramp euler found in slope {angle}"
        eulers[angle] = abs(float(m.group(1)))

    ordered = [eulers[a] for a in SLOPE_ANGLES]
    assert ordered == sorted(ordered), f"slope angles not monotonic: {eulers}"
    # 25 degrees in radians is ~0.4363
    assert eulers[25] == pytest.approx(0.4363, abs=1e-3)


# ===========================================================================
# Paths
# ===========================================================================
def test_no_machine_specific_paths_in_source(repo_root):
    """
    The original blocker: scripts hardcoded /home/user/projects/... which
    exists on no other machine. Committed run logs are historical records and
    are exempt; source code is not.
    """
    # Built at runtime so this file does not match its own pattern.
    needle = "/home/" + "user"
    # tests/ is exempt: it necessarily names the pattern it searches for.
    # paths.py is exempt: its docstring documents the historical path.
    exempt_dirs = {".venv", "__pycache__", "tests"}

    offenders = []
    for py in repo_root.rglob("*.py"):
        if any(part in py.parts for part in exempt_dirs):
            continue
        if py.name == "paths.py":
            continue
        for lineno, line in enumerate(py.read_text().splitlines(), 1):
            if needle in line and not line.lstrip().startswith(("#", "*")):
                offenders.append(f"{py.relative_to(repo_root)}:{lineno}")
    assert not offenders, (
        f"machine-specific absolute paths found in source: {offenders}"
    )


def test_default_model_path_resolves(repo_root):
    import paths
    assert Path(paths.MODEL_PATH).is_absolute()
    assert Path(paths.MODEL_PATH).is_file(), (
        f"default MODEL_PATH does not exist: {paths.MODEL_PATH}"
    )


def test_scene_env_var_override(monkeypatch, repo_root):
    """GO2_SCENE selects a different world without touching any source."""
    monkeypatch.setenv("GO2_SCENE", "go2_stairs_12.xml")
    import importlib
    import paths
    reloaded = importlib.reload(paths)
    try:
        assert reloaded.MODEL_PATH.endswith("go2_stairs_12.xml")
        assert Path(reloaded.MODEL_PATH).is_file()
    finally:
        monkeypatch.delenv("GO2_SCENE", raising=False)
        importlib.reload(paths)


def test_policy_dir_env_var_override(monkeypatch, tmp_path):
    """GO2_POLICY_DIR overrides the default checkpoint location on the host."""
    monkeypatch.setenv("GO2_POLICY_DIR", str(tmp_path))
    import importlib
    import paths
    reloaded = importlib.reload(paths)
    try:
        assert reloaded.POLICY_DIR == str(tmp_path)
        body, adapt = reloaded.policy_paths()
        assert body.endswith("body_latest.jit")
        assert adapt.endswith("adaptation_module_latest.jit")
        assert reloaded.policy_available() is False

        # And the error must be actionable, not a bare "file not found".
        with pytest.raises(FileNotFoundError) as exc:
            reloaded.require_policy()
        message = str(exc.value)
        assert "GO2_POLICY_DIR" in message
        assert "body_latest.jit" in message
        assert "docs/DEPENDENCIES.md" in message
    finally:
        monkeypatch.delenv("GO2_POLICY_DIR", raising=False)
        importlib.reload(paths)


def test_mujoco_requires_absolute_paths(scenes_dir, repo_root):
    """
    Documents a real MuJoCo gotcha: a RELATIVE scene path breaks mesh
    resolution for these scenes, while the absolute one works. Every script
    in the repo wraps its path in abspath() because of this.
    """
    absolute = str((scenes_dir / "go2_flat.xml").resolve())
    mujoco.MjModel.from_xml_path(absolute)  # must not raise

    relative = os.path.relpath(absolute, start=str(repo_root))
    cwd = os.getcwd()
    os.chdir(repo_root)
    try:
        with pytest.raises(ValueError):
            mujoco.MjModel.from_xml_path(relative)
    finally:
        os.chdir(cwd)


# ===========================================================================
# Offscreen rendering
# ===========================================================================
def test_screenshot_resolution_fits_the_framebuffer(repo_root, scenes_dir):
    """
    Regression: 08_view_scene.py requested a 1920x1080 Renderer while every
    scene declared MuJoCo's default 640x480 offscreen framebuffer, so the
    screenshot feature raised on every use.

    mujoco.Renderer REFUSES a request larger than the framebuffer rather than
    resizing, so the model's offwidth/offheight must be enlarged BEFORE any
    Renderer is constructed. This test pins that contract.
    """
    import re

    src = (repo_root / "stage2-go2-mujoco-inference"
           / "08_view_scene.py").read_text()

    m = re.search(r"SHOT_WIDTH,\s*SHOT_HEIGHT\s*=\s*(\d+),\s*(\d+)", src)
    assert m, "08_view_scene.py no longer declares SHOT_WIDTH/SHOT_HEIGHT"
    want_w, want_h = int(m.group(1)), int(m.group(2))

    assert "model.vis.global_.offwidth" in src, (
        "08_view_scene.py must enlarge the offscreen framebuffer before "
        "building a Renderer, or screenshots raise"
    )

    # The default really is smaller than what the script asks for — which is
    # exactly why the enlargement is load-bearing rather than decorative.
    model = mujoco.MjModel.from_xml_path(
        str((scenes_dir / "go2_flat.xml").resolve()))
    assert model.vis.global_.offwidth < want_w

    # Applying the same fix the script applies must make the request legal.
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, want_w)
    model.vis.global_.offheight = max(model.vis.global_.offheight, want_h)
    assert model.vis.global_.offwidth >= want_w
    assert model.vis.global_.offheight >= want_h
