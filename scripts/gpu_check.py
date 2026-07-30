#!/usr/bin/env python3
"""
GPU readiness check — run this after a reboot.

Answers one question: **can this project actually use the GPU right now?**

It checks the four things that must all line up, in the order they fail:

  1. driver      kernel module version == userspace library version
  2. nvidia-smi  the driver responds
  3. torch       built with CUDA, sees the device, can do real arithmetic
  4. rendering   MUJOCO_GL=egl uses the GPU rather than silently falling
                 back to software

The fourth is the sneaky one: EGL happily returns valid frames at software
speed when it cannot load the driver, so "it rendered" proves nothing.

    python scripts/gpu_check.py
    python scripts/gpu_check.py --bench    # also time GPU vs CPU
"""

import argparse
import glob
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_TTY = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def c(code, s):
    return f"\033[{code}m{s}\033[0m" if _TTY else s


OK = c("32", "  OK  ")
BAD = c("31", " FAIL ")
WARN = c("33", " WARN ")
INFO = c("2", " INFO ")

problems = []


def line(mark, name, detail=""):
    print(f"[{mark}] {name:<34} {c('2', detail)}")


# ---------------------------------------------------------------------------
# 1. Driver
# ---------------------------------------------------------------------------
def check_driver():
    loaded = None
    try:
        for ln in Path("/proc/driver/nvidia/version").read_text().splitlines():
            if "NVRM" in ln:
                m = re.search(r"(\d+\.\d+(?:\.\d+)?)", ln)
                loaded = m.group(1) if m else None
                break
    except Exception:
        line(WARN, "nvidia kernel module", "not loaded — no NVIDIA GPU?")
        return False

    userspace = None
    for path in glob.glob("/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.*"):
        m = re.search(r"libnvidia-ml\.so\.(\d[\d.]+)$", path)
        if m:
            userspace = m.group(1)
            break

    if loaded and userspace and loaded != userspace:
        line(BAD, "driver version match",
             f"kernel module {loaded} != userspace {userspace}")
        problems.append(
            f"DRIVER MISMATCH: the running kernel module is {loaded} but the "
            f"userspace libraries are {userspace}.\n"
            "    The driver was upgraded while the machine was running. The "
            "old module cannot\n"
            "    unload because the display server is using it.\n"
            "    FIX: reboot. Nothing to install."
        )
        return False

    line(OK, "driver version match", f"kernel module and userspace both {loaded}")
    return True


def check_smi():
    r = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=name,driver_version,memory.total,memory.used",
         "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        err = (r.stderr or r.stdout).strip().splitlines()
        line(BAD, "nvidia-smi", err[0] if err else "failed")
        problems.append("nvidia-smi cannot talk to the driver.")
        return False
    line(OK, "nvidia-smi", r.stdout.strip().replace("\n", " | "))
    return True


# ---------------------------------------------------------------------------
# 2. torch
# ---------------------------------------------------------------------------
def check_torch():
    try:
        import torch
    except ImportError:
        line(BAD, "torch", "not installed")
        problems.append("torch is not installed — run ./scripts/setup_env.sh")
        return False

    if torch.version.cuda is None:
        line(BAD, "torch CUDA build", f"{torch.__version__} is CPU-only")
        problems.append(
            "torch is a CPU-only build. Install the CUDA wheel:\n"
            "    pip install 'torch==2.13.0+cu130' --index-url "
            "https://download.pytorch.org/whl/cu130"
        )
        return False
    line(OK, "torch CUDA build",
         f"{torch.__version__} (CUDA {torch.version.cuda})")

    if not torch.cuda.is_available():
        line(BAD, "torch sees a GPU", "torch.cuda.is_available() is False")
        problems.append(
            "torch has CUDA support but cannot open the device — almost "
            "always the driver mismatch above."
        )
        return False

    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    cap = torch.cuda.get_device_capability(idx)
    line(OK, "torch sees a GPU",
         f"{props.name} sm_{cap[0]}{cap[1]} {props.total_memory / 1e9:.1f} GB")

    if f"sm_{cap[0]}{cap[1]}" not in torch.cuda.get_arch_list():
        line(WARN, "compiled for this arch",
             f"sm_{cap[0]}{cap[1]} not in {torch.cuda.get_arch_list()} — "
             "will JIT-compile, slow first run")
    else:
        line(OK, "compiled for this arch", f"sm_{cap[0]}{cap[1]} included")

    # Real arithmetic, not just device enumeration.
    try:
        a = torch.randn(512, 512, device="cuda")
        result = (a @ a).sum().item()
        assert result == result  # not NaN
        torch.cuda.synchronize()
        line(OK, "GPU arithmetic", "512x512 matmul returned a finite result")
    except Exception as e:
        line(BAD, "GPU arithmetic", f"{type(e).__name__}: {e}")
        problems.append(f"GPU compute failed: {e}")
        return False

    return True


# ---------------------------------------------------------------------------
# 3. Rendering
# ---------------------------------------------------------------------------
def check_egl():
    """
    EGL returns valid frames at software speed when it cannot load the
    driver, so success alone means nothing. The libEGL warnings on stderr
    are the only reliable tell.
    """
    scene = (REPO_ROOT / "stage2-go2-mujoco-inference" / "scenes"
             / "go2_flat.xml")
    probe = (
        "import mujoco;"
        f"m=mujoco.MjModel.from_xml_path(r'{scene}');"
        "m.vis.global_.offwidth=320;m.vis.global_.offheight=240;"
        "d=mujoco.MjData(m);mujoco.mj_forward(m,d);"
        "r=mujoco.Renderer(m,height=240,width=320);r.update_scene(d);"
        "r.render();r.close();print('RENDERED')"
    )
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                       text=True, env={**os.environ, "MUJOCO_GL": "egl"},
                       timeout=180)
    if r.returncode != 0 or "RENDERED" not in r.stdout:
        tail = (r.stderr or "").strip().splitlines()
        line(BAD, "MUJOCO_GL=egl", tail[-1][:80] if tail else "failed")
        problems.append("EGL offscreen rendering does not work.")
        return False

    if "driver (null)" in r.stderr or "failed to create dri2 screen" in r.stderr:
        line(WARN, "MUJOCO_GL=egl",
             "renders, but SOFTWARE fallback — GPU driver not loaded")
        problems.append(
            "EGL silently fell back to software rendering. It will be no "
            "faster than osmesa until the driver is fixed."
        )
        return False

    line(OK, "MUJOCO_GL=egl", "hardware-accelerated offscreen rendering")
    return True


# ---------------------------------------------------------------------------
# 4. Benchmarks
# ---------------------------------------------------------------------------
def bench():
    import time
    import torch

    print(f"\n{c('1', 'Benchmarks')}")
    print("-" * 70)

    # Matmul: representative of the PPO update.
    for device in ("cpu", "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            continue
        x = torch.randn(4096, 4096, device=device)
        for _ in range(3):
            x @ x
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            x @ x
        if device == "cuda":
            torch.cuda.synchronize()
        el = time.time() - t0
        gflops = 10 * 2 * 4096 ** 3 / el / 1e9
        print(f"  4096^3 matmul   {device:5s}  {el / 10 * 1000:7.1f} ms  "
              f"{gflops:8.0f} GFLOP/s")

    # Rendering, both backends.
    scene = (REPO_ROOT / "stage2-go2-mujoco-inference" / "scenes"
             / "go2_stairs_08.xml")
    probe = (
        "import time,mujoco;"
        f"m=mujoco.MjModel.from_xml_path(r'{scene}');"
        "m.vis.global_.offwidth=1280;m.vis.global_.offheight=720;"
        "d=mujoco.MjData(m);mujoco.mj_forward(m,d);"
        "r=mujoco.Renderer(m,height=720,width=1280);"
        "r.update_scene(d);r.render();"
        "t=time.time();N=20;"
        "[ (r.update_scene(d), r.render()) for _ in range(N)];"
        "print(f'{N/(time.time()-t):.1f}')"
    )
    for backend in ("osmesa", "egl"):
        out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                             text=True, env={**os.environ, "MUJOCO_GL": backend},
                             timeout=300)
        fps = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else "?"
        print(f"  1280x720 render {backend:5s}  {fps:>7} fps")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", action="store_true",
                    help="also time GPU vs CPU matmul and rendering")
    args = ap.parse_args()

    print()
    print(c("1", "=" * 70))
    print(c("1", "  GPU readiness"))
    print(c("1", "=" * 70))

    driver_ok = check_driver()
    if driver_ok:
        check_smi()
    else:
        line(INFO, "nvidia-smi", "skipped — fix the driver first")
    torch_ok = check_torch()
    check_egl()

    print(c("1", "=" * 70))
    if not problems:
        print(c("32", c("1", "  GPU READY — Stage 3 can train on it.")))
        print("\n  Next:  python stage3-go2-training/train.py "
              "--run-name gpu_v1 --device cuda")
        print("         MUJOCO_GL=egl python "
              "stage2-go2-mujoco-inference/experiments/exp1_velocity_sweep.py")
    else:
        print(c("33", c("1", f"  GPU NOT READY — {len(problems)} problem(s):")))
        for p in problems:
            print(c("33", f"\n  - {p}"))
        print(c("2", "\n  The project runs fine on CPU meanwhile; only Stage 3 "
                     "training speed is affected."))
    print(c("1", "=" * 70))
    print()

    if args.bench and torch_ok:
        bench()

    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
