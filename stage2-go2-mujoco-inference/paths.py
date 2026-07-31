"""
Central path + policy resolution for every Stage 2 script.

WHY THIS EXISTS
---------------
The Stage 2 scripts were originally written with absolute paths baked in
(`/home/user/projects/mujoco_menagerie/...`), which only resolved on the
original author's machine. That made the repo un-runnable on any other
computer and impossible to containerize.

This module resolves both external artifacts from environment variables,
falling back to in-repo defaults:

    GO2_SCENE       scene filename inside scenes/       (default go2_flat.xml)
    GO2_MODEL_PATH  full path to a scene XML            (overrides GO2_SCENE)
    GO2_POLICY_DIR  dir holding the two .jit checkpoints
                    (default <repo>/policies/walk-these-ways-go2)

Nothing here changes the physics or the control pipeline — it only decides
which files get opened. See docs/REVERSE-ENGINEERING.md for the full story.

USAGE
-----
    from paths import MODEL_PATH, POLICY_DIR, load_policy
    body_net, adapt_net = load_policy()      # raises a helpful error if absent
"""

import os
import sys
from pathlib import Path

# --- Repo layout ------------------------------------------------------------
STAGE2_DIR = Path(__file__).resolve().parent
REPO_ROOT = STAGE2_DIR.parent
SCENES_DIR = Path(os.environ.get("GO2_SCENES_DIR", STAGE2_DIR / "scenes"))

# --- Scene (robot + terrain) -----------------------------------------------
# Default is the self-contained flat scene committed to this repo. It includes
# scenes/go2_model/go2.xml, which is the mujoco_menagerie Unitree Go2 MJCF
# vendored in-tree (meshes included), so no external checkout is needed.
DEFAULT_SCENE = os.environ.get("GO2_SCENE", "go2_flat.xml")
MODEL_PATH = str(Path(os.environ.get("GO2_MODEL_PATH", SCENES_DIR / DEFAULT_SCENE)))

# --- Policy checkpoints (NOT in this repo — see docs/DEPENDENCIES.md) -------
POLICY_DIR = os.environ.get(
    "GO2_POLICY_DIR", str(REPO_ROOT / "policies" / "walk-these-ways-go2")
)
BODY_JIT = "body_latest.jit"
ADAPT_JIT = "adaptation_module_latest.jit"

_MISSING_POLICY_HELP = """
Cannot find the pretrained walk-these-ways policy checkpoints.

  Looked in : {policy_dir}
  Needed    : {body}
              {adapt}

These are TorchScript exports of a policy trained in Isaac Gym. They are
large binaries and are deliberately NOT committed to this repository, so you
must supply them yourself:

  1. Download them (MIT-licensed Go2 fork commits its pretrained run):

       mkdir -p policies/walk-these-ways-go2
       BASE="https://raw.githubusercontent.com/Teddy-Liao/walk-these-ways-go2/main/runs/gait-conditioned-agility/pretrain-go2/train/142238.667503/checkpoints"
       curl -sL --fail "$BASE/body_latest.jit"              -o policies/walk-these-ways-go2/body_latest.jit
       curl -sL --fail "$BASE/adaptation_module_latest.jit" -o policies/walk-these-ways-go2/adaptation_module_latest.jit

     Or export your own from a walk-these-ways-go2 training run, or from
     Stage 3 (stage3-go2-training/export.py).
  2. Put them anywhere, then point the scripts at that directory:

       export GO2_POLICY_DIR=/absolute/path/to/checkpoints

     or drop them in the default location:

       {default_dir}/

You can still run plenty of this repo without them:
  - Stage 1 in full (all of stage1-rl-fundamentals/)
  - Stage 2 scripts 01-04 (model loading, inspection, PD posing, obs vector)
  - experiments/generate_terrain_scenes.py and check_terrain_visual.py
  - experiments/make_figures.py (regenerates all 4 data figures from the
    committed CSVs)

See docs/DEPENDENCIES.md for details.
""".strip()


def policy_paths():
    """Return (body_jit_path, adaptation_jit_path) as strings."""
    return (
        str(Path(POLICY_DIR) / BODY_JIT),
        str(Path(POLICY_DIR) / ADAPT_JIT),
    )


def policy_available():
    """True if both checkpoint files exist on disk."""
    return all(Path(p).is_file() for p in policy_paths())


def require_policy():
    """Raise FileNotFoundError with actionable guidance if checkpoints absent."""
    if policy_available():
        return
    body, adapt = policy_paths()
    raise FileNotFoundError(
        _MISSING_POLICY_HELP.format(
            policy_dir=POLICY_DIR,
            body=body,
            adapt=adapt,
            default_dir=REPO_ROOT / "policies" / "walk-these-ways-go2",
        )
    )


def load_policy(eval_mode=True):
    """
    Load both TorchScript networks.

    Returns (body_net, adapt_net). The body net maps
    [obs history (2100) + env latent (2)] -> 12 joint deltas; the adaptation
    module maps [obs history (2100)] -> 2-dim env latent.
    """
    import torch  # imported lazily so scripts 01-04 don't need torch

    require_policy()
    body_path, adapt_path = policy_paths()
    body_net = torch.jit.load(body_path)
    adapt_net = torch.jit.load(adapt_path)
    if eval_mode:
        body_net.eval()
        adapt_net.eval()
    return body_net, adapt_net


def add_stage2_to_syspath():
    """Let scripts in subdirs (e.g. experiments/) import this module."""
    if str(STAGE2_DIR) not in sys.path:
        sys.path.insert(0, str(STAGE2_DIR))


if __name__ == "__main__":
    print(f"REPO_ROOT   : {REPO_ROOT}")
    print(f"SCENES_DIR  : {SCENES_DIR}")
    print(f"MODEL_PATH  : {MODEL_PATH}  (exists: {Path(MODEL_PATH).is_file()})")
    print(f"POLICY_DIR  : {POLICY_DIR}")
    for p in policy_paths():
        print(f"  {'OK  ' if Path(p).is_file() else 'MISS'} {p}")
