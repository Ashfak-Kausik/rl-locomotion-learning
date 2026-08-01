#!/usr/bin/env python3
"""
Detect, record and compare the hardware this project runs on.

Why this exists: Stage 3 training results are only comparable if you know what
machine produced them. A run at 1,150 steps/s on an 8-core CPU with an RTX 3050
is a different experiment from the same script on a 32-core box with a 4090,
and until now nothing in the repo recorded which was which. Worse, the GPU
tuning in `stage3-go2-training/config.py` was written against one specific card
and silently applied to every other one.

So: one command detects everything that matters, and `--save` commits it as
`hardware/profiles/<hostname>.json`. Contributors run it once; after that the
repo knows what it is running on and `tune_profiles.py` can pick sane batch
sizes instead of guessing.

Deliberately vendor-neutral. NVIDIA is what this repo was developed on, but
Intel Arc (XPU), AMD (ROCm) and Apple (MPS) are all detected, because a
contributor on any of them currently gets a silent CPU fallback with no
warning -- the single worst failure mode for a 4-hour training job.

Usage
-----
    python scripts/hw_profile.py                # detect and print
    python scripts/hw_profile.py --save         # write hardware/profiles/
    python scripts/hw_profile.py --list         # every recorded profile
    python scripts/hw_profile.py --check        # do I match a known profile?
    python scripts/hw_profile.py --json         # machine-readable
"""

import argparse
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILE_DIR = REPO_ROOT / "hardware" / "profiles"

_TTY = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def c(code, s):
    return f"\033[{code}m{s}\033[0m" if _TTY else s


BOLD = lambda s: c("1", s)      # noqa: E731
DIM = lambda s: c("2", s)       # noqa: E731
GREEN = lambda s: c("32", s)    # noqa: E731
YELLOW = lambda s: c("33", s)   # noqa: E731


def _run(cmd, timeout=15):
    """Run a command, return stdout or None. Never raises."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


# ===========================================================================
# CPU / memory
# ===========================================================================
def detect_cpu():
    info = {
        "model": platform.processor() or "unknown",
        "arch": platform.machine(),
        "cores_physical": None,
        "cores_logical": os.cpu_count(),
        "max_mhz": None,
    }

    # /proc/cpuinfo is the most reliable source of the marketing name on
    # Linux; platform.processor() often returns just "x86_64".
    try:
        text = Path("/proc/cpuinfo").read_text()
        m = re.search(r"^model name\s*:\s*(.+)$", text, re.M)
        if m:
            info["model"] = m.group(1).strip()
        cores = re.search(r"^cpu cores\s*:\s*(\d+)$", text, re.M)
        if cores:
            info["cores_physical"] = int(cores.group(1))
    except Exception:
        pass

    lscpu = _run(["lscpu"])
    if lscpu:
        for key, field in (("CPU max MHz", "max_mhz"),):
            m = re.search(rf"^{key}:\s*(\S+)", lscpu, re.M)
            if m:
                try:
                    info[field] = float(m.group(1))
                except ValueError:
                    pass

    if info["cores_physical"] is None:
        info["cores_physical"] = info["cores_logical"]

    # SMT/Hyper-Threading changes how many MuJoCo envs are worth running:
    # rollout collection is CPU-bound and SMT siblings do not help it much.
    info["smt"] = (info["cores_logical"] or 0) > (info["cores_physical"] or 0)
    return info


def detect_memory():
    mem = {"total_gb": None, "available_gb": None, "swap_gb": None}
    try:
        text = Path("/proc/meminfo").read_text()

        def kb(field):
            m = re.search(rf"^{field}:\s*(\d+) kB", text, re.M)
            return int(m.group(1)) if m else None

        for field, key in (("MemTotal", "total_gb"),
                           ("MemAvailable", "available_gb"),
                           ("SwapTotal", "swap_gb")):
            v = kb(field)
            if v is not None:
                mem[key] = round(v / 1024 / 1024, 1)
    except Exception:
        pass
    return mem


# ===========================================================================
# GPUs — every vendor, because a silent CPU fallback costs hours
# ===========================================================================
def detect_nvidia():
    if not shutil.which("nvidia-smi"):
        return []
    out = _run(["nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader,nounits"])
    if not out:
        return []
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        gpus.append({
            "vendor": "nvidia",
            "backend": "cuda",
            "name": parts[0],
            "driver": parts[1],
            "vram_gb": round(float(parts[2]) / 1024, 1),
            "compute_cap": parts[3] if len(parts) > 3 else None,
        })
    return gpus


def detect_intel():
    """
    Intel Arc / integrated graphics. Reported whether or not torch can use it,
    because "you have an Arc card but a CPU-only torch" is exactly the state a
    contributor needs told about.
    """
    gpus = []

    # xpu-smi is the Arc equivalent of nvidia-smi. Rarely installed, so
    # lspci is the fallback that always works.
    lspci = _run(["lspci", "-nn"])
    if lspci:
        for line in lspci.splitlines():
            if not re.search(r"VGA|3D controller|Display", line, re.I):
                continue
            if "Intel" not in line:
                continue
            m = re.search(r":\s*(.+?)\s*\[[0-9a-f]{4}:[0-9a-f]{4}\]", line)
            name = m.group(1).strip() if m else line.strip()
            # Arc discrete cards are the ones worth training on; integrated
            # UHD/Iris is detected but flagged as not useful for Stage 3.
            discrete = bool(re.search(r"\bArc\b|DG2|BMG|Battlemage|Alchemist",
                                      name, re.I))
            gpus.append({
                "vendor": "intel",
                "backend": "xpu",
                "name": name,
                "driver": None,
                "vram_gb": None,
                "discrete": discrete,
            })

    smi = _run(["xpu-smi", "discovery", "-j"])
    if smi:
        try:
            data = json.loads(smi)
            for dev in data.get("device_list", []):
                for g in gpus:
                    if g["vendor"] == "intel":
                        g["driver"] = dev.get("driver_version")
                        mem = dev.get("memory_physical_size_byte")
                        if mem:
                            g["vram_gb"] = round(int(mem) / 1e9, 1)
                        break
        except Exception:
            pass

    return gpus


def detect_amd():
    if not shutil.which("rocm-smi"):
        return []
    out = _run(["rocm-smi", "--showproductname", "--csv"])
    if not out:
        return []
    gpus = []
    for line in out.splitlines()[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[1]:
            gpus.append({
                "vendor": "amd", "backend": "rocm", "name": parts[1],
                "driver": None, "vram_gb": None,
            })
    return gpus


def detect_apple():
    if platform.system() != "Darwin":
        return []
    name = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "Apple Silicon"
    if "Apple" not in name:
        return []
    return [{"vendor": "apple", "backend": "mps", "name": name,
             "driver": None, "vram_gb": None}]


def detect_gpus():
    return (detect_nvidia() + detect_intel() + detect_amd() + detect_apple())


# ===========================================================================
# torch — what the training script will ACTUALLY get
# ===========================================================================
def detect_torch():
    """
    The gap between "you have a GPU" and "torch can use your GPU" is where
    every silent CPU fallback lives, so report both sides of it.
    """
    info = {"installed": False}
    try:
        import torch
    except ImportError:
        return info

    info.update({
        "installed": True,
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "xpu_available": bool(getattr(torch, "xpu", None)
                              and torch.xpu.is_available()),
        "mps_available": bool(getattr(torch.backends, "mps", None)
                              and torch.backends.mps.is_available()),
        "threads": torch.get_num_threads(),
    })

    if info["cuda_available"]:
        idx = torch.cuda.current_device()
        cap = torch.cuda.get_device_capability(idx)
        info["device_name"] = torch.cuda.get_device_name(idx)
        info["compute_cap"] = f"sm_{cap[0]}{cap[1]}"
        info["arch_list"] = list(torch.cuda.get_arch_list())
        # A card not in arch_list still works, but JIT-compiles kernels on
        # first use -- minutes of apparent hang at the start of training.
        info["arch_prebuilt"] = info["compute_cap"] in info["arch_list"]
    elif info["xpu_available"]:
        info["device_name"] = torch.xpu.get_device_name(0)

    return info


def effective_backend(profile):
    """Which torch device Stage 3 will really land on, given this machine."""
    t = profile["torch"]
    if not t.get("installed"):
        return "none"
    for key, backend in (("cuda_available", "cuda"),
                         ("xpu_available", "xpu"),
                         ("mps_available", "mps")):
        if t.get(key):
            return backend
    return "cpu"


# ===========================================================================
# Profile assembly
# ===========================================================================
def build_profile():
    try:
        osr = dict(
            ln.split("=", 1)
            for ln in Path("/etc/os-release").read_text().splitlines()
            if "=" in ln
        )
        os_name = osr.get("PRETTY_NAME", "").strip('"')
    except Exception:
        os_name = f"{platform.system()} {platform.release()}"

    profile = {
        "hostname": socket.gethostname(),
        "os": os_name,
        "kernel": platform.release(),
        "python": platform.python_version(),
        "cpu": detect_cpu(),
        "memory": detect_memory(),
        "gpus": detect_gpus(),
        "torch": detect_torch(),
    }
    profile["backend"] = effective_backend(profile)
    return profile


def profile_key(profile):
    """
    A short identifier for the machine class, used to match against tuning
    profiles. Keyed on the training-relevant facts only -- the same card in a
    different case is the same profile.
    """
    gpus = profile["gpus"]
    if not gpus:
        return f"cpu-{profile['cpu']['cores_logical']}t"
    g = gpus[0]
    name = re.sub(r"[^a-z0-9]+", "-",
                  (g["name"] or "gpu").lower()).strip("-")
    vram = f"-{g['vram_gb']:g}gb" if g.get("vram_gb") else ""
    return f"{name}{vram}"


# ===========================================================================
# Reporting
# ===========================================================================
def print_profile(profile):
    cpu, mem, t = profile["cpu"], profile["memory"], profile["torch"]

    print()
    print(BOLD("=" * 74))
    print(BOLD(f"  hardware profile — {profile['hostname']}"))
    print(BOLD("=" * 74))

    print(f"\n{BOLD('SYSTEM')}")
    print(f"  os                  {profile['os']}")
    print(f"  kernel              {profile['kernel']}")
    print(f"  python              {profile['python']}")

    print(f"\n{BOLD('CPU')}")
    print(f"  model               {cpu['model']}")
    smt = "SMT/HT on" if cpu["smt"] else "no SMT"
    print(f"  cores               {cpu['cores_physical']} physical / "
          f"{cpu['cores_logical']} logical  ({smt})")
    if cpu["max_mhz"]:
        print(f"  max clock           {cpu['max_mhz'] / 1000:.2f} GHz")

    print(f"\n{BOLD('MEMORY')}")
    print(f"  ram                 {mem['total_gb']} GB "
          f"({mem['available_gb']} GB available)")
    print(f"  swap                {mem['swap_gb']} GB")

    print(f"\n{BOLD('GPU')}")
    if not profile["gpus"]:
        print(DIM("  none detected — Stages 1-2 are CPU-only anyway"))
    for g in profile["gpus"]:
        vram = f"{g['vram_gb']} GB" if g.get("vram_gb") else "? GB"
        extra = f"  {g['compute_cap']}" if g.get("compute_cap") else ""
        integrated = ("  " + YELLOW("integrated — not useful for training")
                      if g.get("discrete") is False else "")
        print(f"  [{g['vendor']:6}] {g['name']}")
        print(DIM(f"           {vram}  driver {g['driver'] or '?'}"
                  f"{extra}{integrated}"))

    print(f"\n{BOLD('TORCH')}")
    if not t.get("installed"):
        print(YELLOW("  not installed — run ./scripts/setup_env.sh"))
    else:
        flavour = f"CUDA {t['cuda_build']}" if t.get("cuda_build") else "CPU-only"
        print(f"  version             {t['version']}  ({flavour})")
        print(f"  threads             {t['threads']}")
        if t.get("device_name"):
            print(f"  device              {t['device_name']}")
        if t.get("arch_prebuilt") is False:
            print(YELLOW(f"  ! {t['compute_cap']} is not in this wheel's "
                         f"arch list {t['arch_list']}"))
            print(DIM("    kernels will JIT-compile — a slow first run, "
                      "not a failure"))

    backend = profile["backend"]
    print(f"\n{BOLD('EFFECTIVE TRAINING BACKEND')}")
    if backend in ("cuda", "xpu", "mps"):
        print(GREEN(f"  {backend}  — Stage 3 will train on the GPU"))
    elif backend == "cpu" and profile["gpus"]:
        print(YELLOW("  cpu  — a GPU IS present but torch cannot use it"))
        vendors = {g["vendor"] for g in profile["gpus"]}
        if "intel" in vendors:
            print(DIM("    Intel Arc needs the XPU wheel:"))
            print(DIM("      ./scripts/setup_env.sh --xpu"))
        if "nvidia" in vendors:
            print(DIM("    NVIDIA needs the CUDA wheel:"))
            print(DIM("      ./scripts/setup_env.sh --cuda"))
    else:
        print(f"  {backend}")

    print(f"\n  profile key         {profile_key(profile)}")
    print(BOLD("=" * 74))
    print()


def cmd_save(profile):
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    path = PROFILE_DIR / f"{profile['hostname']}.json"
    existed = path.is_file()
    path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    verb = "updated" if existed else "saved"
    print(f"{GREEN(verb)} {path.relative_to(REPO_ROOT)}")
    print(DIM("  commit it so other contributors' runs are comparable to "
              "yours"))
    return 0


def load_profiles():
    if not PROFILE_DIR.is_dir():
        return []
    out = []
    for p in sorted(PROFILE_DIR.glob("*.json")):
        try:
            out.append((p, json.loads(p.read_text())))
        except Exception as e:
            print(YELLOW(f"  ! {p.name} is not readable: {e}"))
    return out


def cmd_list():
    profiles = load_profiles()
    if not profiles:
        print(YELLOW(f"no profiles recorded in "
                     f"{PROFILE_DIR.relative_to(REPO_ROOT)}"))
        print(DIM("  record this machine with: "
                  "python scripts/hw_profile.py --save"))
        return 0

    print(f"\n{BOLD('recorded hardware profiles')}")
    print(DIM("-" * 74))
    for path, p in profiles:
        gpu = p["gpus"][0]["name"] if p.get("gpus") else "no GPU"
        vram = ""
        if p.get("gpus") and p["gpus"][0].get("vram_gb"):
            vram = f" {p['gpus'][0]['vram_gb']}GB"
        cores = p["cpu"]["cores_logical"]
        print(f"  {BOLD(p['hostname']):<28} {gpu}{vram}")
        print(DIM(f"    {cores}t cpu, {p['memory']['total_gb']}GB ram, "
                  f"backend={p.get('backend', '?')}, key={profile_key(p)}"))
    print()
    return 0


def cmd_check(profile):
    """Does this machine match something already recorded?"""
    key = profile_key(profile)
    recorded = load_profiles()
    matches = [(p, d) for p, d in recorded if profile_key(d) == key]

    print(f"\nthis machine: {BOLD(key)}")
    if matches:
        names = ", ".join(d["hostname"] for _, d in matches)
        print(GREEN(f"  matches recorded profile(s): {names}"))
        return 0

    print(YELLOW("  NEW — no recorded profile matches this hardware"))
    print(DIM("  record it:  python scripts/hw_profile.py --save"))
    if recorded:
        print(DIM(f"  known keys: "
                  f"{', '.join(sorted({profile_key(d) for _, d in recorded}))}"))
    return 1


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save", action="store_true",
                    help="write hardware/profiles/<hostname>.json")
    ap.add_argument("--list", action="store_true",
                    help="show every recorded profile")
    ap.add_argument("--check", action="store_true",
                    help="does this machine match a recorded profile?")
    ap.add_argument("--json", action="store_true",
                    help="emit the profile as JSON")
    args = ap.parse_args()

    if args.list:
        return cmd_list()

    profile = build_profile()

    if args.json:
        print(json.dumps(profile, indent=2, sort_keys=True))
        return 0
    if args.check:
        return cmd_check(profile)

    print_profile(profile)
    if args.save:
        return cmd_save(profile)
    return 0


if __name__ == "__main__":
    sys.exit(main())
