#!/usr/bin/env python3
"""
Environment verifier for rl-locomotion-learning.

Checks — and reports, without changing anything — every third-party
dependency this repo needs, in four layers:

  1. System   : OS, Python version, native libs, GPU/driver, Docker
  2. Python   : importable packages + versions vs. requirements.txt
  3. Assets   : in-repo scenes/meshes, out-of-repo policy checkpoints
  4. Runtime  : does MuJoCo actually load a model? does a Gym env build?

Exit code 0 = everything needed for the full pipeline is present.
Exit code 1 = at least one REQUIRED item is missing (details printed).
Optional items never fail the run; they are reported as WARN.

Usage:
    python3 scripts/check_env.py            # full report
    python3 scripts/check_env.py --quiet    # only problems + summary
    python3 scripts/check_env.py --json     # machine-readable
"""

import argparse
import importlib
import importlib.metadata as md
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- terminal colours (auto-disabled when not a TTY) ------------------------
_TTY = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def c(code, s):
    return f"\033[{code}m{s}\033[0m" if _TTY else s


BOLD = lambda s: c("1", s)      # noqa: E731
GREEN = lambda s: c("32", s)    # noqa: E731
YELLOW = lambda s: c("33", s)   # noqa: E731
RED = lambda s: c("31", s)      # noqa: E731
DIM = lambda s: c("2", s)       # noqa: E731

RESULTS = []  # (layer, name, status, detail, required)
STATUS_MARK = {
    "ok": GREEN("  OK  "),
    "warn": YELLOW(" WARN "),
    "fail": RED(" FAIL "),
    "info": DIM(" INFO "),
}


def record(layer, name, status, detail="", required=True):
    RESULTS.append(
        {"layer": layer, "name": name, "status": status,
         "detail": detail, "required": required}
    )
    return status == "ok"


# ============================================================================
# LAYER 1 — system
# ============================================================================
# Native shared libraries. MuJoCo's `launch_passive` viewer needs GLFW + an
# OpenGL driver; headless rendering (mujoco.Renderer) needs EGL or OSMesa.
APT_LIBS = {
    "libglfw3": ("MuJoCo interactive viewer window", False),
    "libgl1": ("OpenGL runtime — required by MuJoCo rendering", True),
    "libegl1": ("headless GPU rendering backend (MUJOCO_GL=egl)", False),
    "libosmesa6": ("headless software rendering (MUJOCO_GL=osmesa)", False),
}

# Checked by capability rather than by package name, so the result is correct
# on any distro and inside slim container images where the apt package that
# would normally provide a thing was never installed as such.
CAPABILITIES = {
    "swig": ("Box2D binding generator (LunarLander-v3)", False),
    "ffmpeg": ("encodes demo GIFs/MP4s from rendered frames", False),
    "patchelf": ("occasionally needed by MuJoCo/Gym native deps", False),
    "git": ("version control", True),
}


def check_system():
    layer = "system"

    # OS
    try:
        osr = dict(
            line.split("=", 1)
            for line in Path("/etc/os-release").read_text().splitlines()
            if "=" in line
        )
        pretty = osr.get("PRETTY_NAME", "").strip('"')
    except Exception:
        pretty = f"{platform.system()} {platform.release()}"
    record(layer, "operating system", "info", pretty)

    # Python — the repo targets 3.10+; 3.10/3.11/3.12 all resolve cleanly.
    v = sys.version_info
    pv = f"{v.major}.{v.minor}.{v.micro}"
    if v >= (3, 10):
        record(layer, "python >= 3.10", "ok", f"{pv} at {sys.executable}")
    else:
        record(layer, "python >= 3.10", "fail",
               f"found {pv}; mujoco/gymnasium/SB3 all require >= 3.10")

    # A container IS the isolation, so venv/docker checks are noise in there.
    in_container = (
        Path("/.dockerenv").exists()
        or os.environ.get("container") is not None
    )
    if in_container:
        record(layer, "execution context", "info",
               "inside a container — venv and docker checks skipped",
               required=False)

    # Virtualenv — installing into the system interpreter is a footgun on
    # Debian/Ubuntu derivatives (PEP 668 externally-managed-environment).
    if not in_container:
        in_venv = sys.prefix != sys.base_prefix
        record(layer, "running inside virtualenv",
               "ok" if in_venv else "warn",
               str(sys.prefix) if in_venv else
               "system interpreter — use scripts/setup_env.sh to create .venv",
               required=False)

    # C toolchain — needed to build Box2D from source when no wheel exists.
    # Check for a working compiler, not for a distro package name.
    cc = next((x for x in ("cc", "gcc", "clang") if shutil.which(x)), None)
    record(layer, "C/C++ compiler",
           "ok" if cc else "warn",
           f"{cc} at {shutil.which(cc)}" if cc else
           "none found — only needed if Box2D has no prebuilt wheel "
           "for your platform (install build-essential / gcc)",
           required=False)

    # Python headers — same story: presence of Python.h is the real question.
    import sysconfig
    py_h = Path(sysconfig.get_paths()["include"]) / "Python.h"
    record(layer, "python headers (Python.h)",
           "ok" if py_h.is_file() else "warn",
           str(py_h) if py_h.is_file() else
           f"missing at {py_h} — needed only to compile Box2D from source "
           "(install python3-dev / python3-devel)",
           required=False)

    # Shared libraries queried through the package manager where available.
    have_dpkg = shutil.which("dpkg-query") is not None
    for pkg, (why, required) in APT_LIBS.items():
        if have_dpkg:
            found = subprocess.run(
                ["dpkg-query", "-W", "-f=${Status}", pkg],
                capture_output=True, text=True,
            ).stdout.strip().endswith("installed")
            record(layer, f"lib: {pkg}",
                   "ok" if found else ("fail" if required else "warn"),
                   why, required=required)
        else:
            record(layer, f"lib: {pkg}", "info",
                   f"{why} — cannot verify (no dpkg); "
                   "layer 4 proves whether rendering actually works",
                   required=False)

    # Executables on PATH.
    for exe, (why, required) in CAPABILITIES.items():
        found = shutil.which(exe)
        record(layer, f"bin: {exe}",
               "ok" if found else ("fail" if required else "warn"),
               f"{found} — {why}" if found else why, required=required)

    # Docker (optional — only for the containerised path)
    if in_container:
        pass
    elif shutil.which("docker"):
        ver = subprocess.run(["docker", "--version"], capture_output=True,
                             text=True).stdout.strip()
        compose = subprocess.run(["docker", "compose", "version"],
                                 capture_output=True, text=True)
        detail = ver
        if compose.returncode == 0:
            detail += f" | {compose.stdout.strip().splitlines()[0]}"
        else:
            detail += " | compose plugin MISSING"
        record(layer, "docker", "ok", detail, required=False)
    else:
        record(layer, "docker", "warn",
               "not installed — needed only for the container workflow",
               required=False)

    # GPU — genuinely optional. This project is CPU-only by design; a GPU
    # matters only for Stage 3 (MJX / mujoco_playground training).
    if shutil.which("nvidia-smi"):
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        if r.returncode == 0 and r.stdout.strip():
            record("system", "nvidia gpu", "ok", r.stdout.strip(),
                   required=False)
        else:
            record("system", "nvidia gpu", "warn",
                   "nvidia-smi present but failing "
                   f"({(r.stderr or r.stdout).strip().splitlines()[0] if (r.stderr or r.stdout).strip() else 'unknown'})"
                   " — driver/library mismatch usually needs a reboot. "
                   "Not required: Stages 1-2 are CPU-only.",
                   required=False)
    else:
        record("system", "nvidia gpu", "info",
               "none detected — fine, Stages 1-2 are CPU-only", required=False)


# ============================================================================
# LAYER 2 — Python packages
# ============================================================================
# (import name, distribution name, why, required)
PY_PACKAGES = [
    ("numpy", "numpy", "array maths everywhere", True),
    ("mujoco", "mujoco", "physics engine — all of Stage 2", True),
    ("torch", "torch", "TorchScript policy inference", True),
    ("matplotlib", "matplotlib", "experiments/make_figures.py", True),
    ("imageio", "imageio", "08_view_scene.py screenshot capture", False),
    ("gymnasium", "gymnasium", "Stage 1 environments", True),
    ("stable_baselines3", "stable-baselines3", "Stage 1 PPO", True),
    ("tensorboard", "tensorboard", "Stage 1 training curves", False),
    ("tqdm", "tqdm", "SB3 progress_bar=True", False),
    ("rich", "rich", "SB3 progress_bar=True", False),
    ("Box2D", "box2d", "LunarLander-v3 physics", False),
]


def check_python_packages():
    layer = "python"
    for import_name, dist_name, why, required in PY_PACKAGES:
        try:
            importlib.import_module(import_name)
        except Exception as e:
            record(layer, dist_name,
                   "fail" if required else "warn",
                   f"{why} — import failed: {type(e).__name__}: {e}",
                   required=required)
            continue
        try:
            ver = md.version(dist_name)
        except Exception:
            ver = "?"
        record(layer, dist_name, "ok", f"{ver} — {why}", required=required)

    # Torch build flavour: CUDA wheels are ~2.5 GB and pointless on a
    # CPU-only box, so surface which one is installed.
    try:
        import torch
        flavour = (f"CUDA {torch.version.cuda}"
                   if getattr(torch.version, "cuda", None) else "CPU-only")
        record(layer, "torch build", "info",
               f"{flavour} | cuda available: {torch.cuda.is_available()} "
               f"| threads: {torch.get_num_threads()}", required=False)
    except Exception:
        pass


# ============================================================================
# LAYER 3 — data assets
# ============================================================================
def check_assets():
    layer = "assets"
    stage2 = REPO_ROOT / "stage2-go2-mujoco-inference"
    scenes = stage2 / "scenes"

    # In-repo: the vendored Go2 MJCF. Committed, so this should always pass.
    go2_xml = scenes / "go2_model" / "go2.xml"
    record(layer, "go2 MJCF (vendored)",
           "ok" if go2_xml.is_file() else "fail",
           str(go2_xml.relative_to(REPO_ROOT)) if go2_xml.is_file()
           else f"missing {go2_xml} — repo checkout incomplete")

    meshes = list((scenes / "go2_model").glob("*.obj"))
    record(layer, "go2 collision/visual meshes",
           "ok" if len(meshes) >= 16 else "warn",
           f"{len(meshes)} .obj files found (expect 16)")

    scene_xmls = sorted(p.name for p in scenes.glob("go2_*.xml"))
    record(layer, "terrain scenes",
           "ok" if len(scene_xmls) >= 11 else "warn",
           f"{len(scene_xmls)} scenes: {', '.join(scene_xmls)}"
           if scene_xmls else "none — run experiments/generate_terrain_scenes.py")

    # Out-of-repo: the pretrained policy. This is the one dependency that
    # cannot be auto-installed, so make the failure loud and specific.
    sys.path.insert(0, str(stage2))
    try:
        import paths as go2_paths
        body, adapt = go2_paths.policy_paths()
        if go2_paths.policy_available():
            size_mb = sum(Path(p).stat().st_size for p in (body, adapt)) / 1e6
            record(layer, "walk-these-ways checkpoints", "ok",
                   f"{go2_paths.POLICY_DIR} ({size_mb:.1f} MB)")
        else:
            record(layer, "walk-these-ways checkpoints", "warn",
                   f"NOT FOUND in {go2_paths.POLICY_DIR}. Cannot be installed "
                   "automatically — supply body_latest.jit + "
                   "adaptation_module_latest.jit and set GO2_POLICY_DIR. "
                   "Blocks scripts 05-08 and experiments 1-3 ONLY; "
                   "everything else still runs. See docs/DEPENDENCIES.md.",
                   required=False)
    except Exception as e:
        record(layer, "walk-these-ways checkpoints", "warn",
               f"path resolution failed: {e}", required=False)

    # Committed experiment data — lets make_figures.py run with no policy.
    csvs = list((stage2 / "experiments" / "results").glob("*.csv"))
    record(layer, "experiment result CSVs",
           "ok" if len(csvs) >= 3 else "warn",
           f"{len(csvs)} CSVs — enough to regenerate all 4 data figures "
           "without the policy" if csvs else "missing",
           required=False)


# ============================================================================
# LAYER 4 — runtime smoke tests
# ============================================================================
def check_runtime():
    layer = "runtime"

    # Can MuJoCo parse the vendored model and step physics?
    try:
        import mujoco
        scene = (REPO_ROOT / "stage2-go2-mujoco-inference" / "scenes"
                 / "go2_flat.xml")
        m = mujoco.MjModel.from_xml_path(str(scene))
        d = mujoco.MjData(m)
        mujoco.mj_step(m, d)
        record(layer, "mujoco loads go2_flat.xml", "ok",
               f"nq={m.nq} nv={m.nv} nu={m.nu} dt={m.opt.timestep}s "
               f"({1 / m.opt.timestep:.0f} Hz) — expect nq=19 nv=18 nu=12")
    except Exception as e:
        record(layer, "mujoco loads go2_flat.xml", "fail",
               f"{type(e).__name__}: {e}")

    # Does the GL stack work? Only matters for viewer/rendering scripts.
    gl = os.environ.get("MUJOCO_GL", "(unset — defaults to glfw)")
    record(layer, "MUJOCO_GL backend", "info",
           f"{gl}  [set MUJOCO_GL=osmesa or egl for headless rendering]",
           required=False)

    # Gymnasium: CartPole is pure-Python; LunarLander needs the Box2D build.
    for env_id, required in (("CartPole-v1", True), ("LunarLander-v3", False)):
        try:
            import gymnasium as gym
            env = gym.make(env_id)
            env.reset(seed=0)
            env.close()
            record(layer, f"gym env {env_id}", "ok", "builds and resets",
                   required=required)
        except Exception as e:
            record(layer, f"gym env {env_id}",
                   "fail" if required else "warn",
                   f"{type(e).__name__}: {e}"
                   + ("  [needs swig + build-essential + python3-dev, then "
                      "reinstall gymnasium[box2d]]" if "Box2D" in str(e)
                      or "box2d" in str(e) else ""),
                   required=required)


# ============================================================================
# report
# ============================================================================
def print_report(quiet=False):
    print()
    print(BOLD("=" * 78))
    print(BOLD("  rl-locomotion-learning — environment check"))
    print(BOLD("=" * 78))

    layers = {
        "system": "1. SYSTEM  (OS, interpreter, native libs, docker, gpu)",
        "python": "2. PYTHON  (packages from requirements.txt)",
        "assets": "3. ASSETS  (robot model, terrain scenes, policy weights)",
        "runtime": "4. RUNTIME (does it actually work?)",
    }
    for layer, title in layers.items():
        rows = [r for r in RESULTS if r["layer"] == layer]
        if not rows:
            continue
        shown = [r for r in rows
                 if not quiet or r["status"] in ("fail", "warn")]
        if quiet and not shown:
            continue
        print(f"\n{BOLD(title)}")
        print(DIM("-" * 78))
        for r in shown:
            print(f"[{STATUS_MARK[r['status']]}] {r['name']:<32} "
                  f"{DIM(r['detail'])}")

    fails = [r for r in RESULTS if r["status"] == "fail"]
    warns = [r for r in RESULTS if r["status"] == "warn"]

    print()
    print(BOLD("=" * 78))
    if not fails and not warns:
        print(GREEN(BOLD("  READY — full pipeline can run.")))
    elif not fails:
        print(YELLOW(BOLD(f"  READY with {len(warns)} optional item(s) missing.")))
        for r in warns:
            print(YELLOW(f"    - {r['name']}: {r['detail'][:100]}"))
    else:
        print(RED(BOLD(f"  BLOCKED — {len(fails)} required item(s) missing, "
                       f"{len(warns)} optional.")))
        for r in fails:
            print(RED(f"    - {r['name']}: {r['detail'][:100]}"))
        print()
        print("  Fix everything installable with:  " + BOLD("./scripts/setup_env.sh"))
    print(BOLD("=" * 78))
    print()
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true",
                    help="only show problems")
    ap.add_argument("--json", action="store_true",
                    help="emit machine-readable JSON")
    args = ap.parse_args()

    check_system()
    check_python_packages()
    check_assets()
    check_runtime()

    if args.json:
        fails = [r for r in RESULTS if r["status"] == "fail"]
        print(json.dumps({"ready": not fails, "checks": RESULTS}, indent=2))
        return 1 if fails else 0
    return print_report(quiet=args.quiet)


if __name__ == "__main__":
    sys.exit(main())
