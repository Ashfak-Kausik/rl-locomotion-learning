"""
Per-machine PPO tuning profiles.

The problem this replaces: `tune_for_device()` used to apply ONE set of
numbers -- num_envs=32, rollout_steps=64, minibatch_size=512 -- to any device
that reported `cuda`. Those numbers were tuned by measurement on an RTX 3050
with an 8-core i7-9700K, and they are wrong everywhere else. A 4 GB card OOMs
on them; a 4090 with 32 cores is left idle by them; an Intel Arc card never
reached them at all because `resolve_device()` did not know `xpu` existed and
silently fell back to CPU.

So the numbers move here, keyed on backend and VRAM, each with the reasoning
attached and an honest flag for whether anyone has actually benchmarked that
configuration or whether it is a conservative guess.

**Measured profiles are the only ones with authority.** A profile marked
`measured=False` is a starting point that will not crash, not a recommendation.
If you run on one, benchmark it and send the numbers back:

    python scripts/hw_profile.py --save
    python stage3-go2-training/train.py --run-name tune_probe --timesteps 200000

and record the steps/s the log reports.

Sizing rules these profiles follow
----------------------------------
1. `minibatch_size` MUST be strictly smaller than
   `num_envs * rollout_steps`, or "minibatching" silently becomes full-batch
   gradient descent. This caused a real multi-hour failure -- see the comment
   on `tune_profile` in config.py. `TuneProfile.validate()` enforces it, so
   the mistake cannot be reintroduced by a new profile. Every profile here
   targets 4 minibatches per epoch.
2. `num_envs` is bounded by CPU, not GPU. Rollout collection steps MuJoCo
   sequentially in Python, so it is single-core-bound; past roughly
   4x physical cores you are adding latency, not throughput.
3. `tf32` stays off unless measured. It introduces a numerical floor in the
   KL estimate that the target_kl guard cannot distinguish from a real policy
   change, which silently killed a 10M-step run.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TuneProfile:
    name: str
    backend: str                 # "cuda" | "xpu" | "mps" | "cpu"
    num_envs: int
    rollout_steps: int
    minibatch_size: int
    tf32: bool
    torch_threads: int | None    # None = leave torch's default alone
    measured: bool
    rationale: str

    @property
    def steps_per_update(self):
        return self.num_envs * self.rollout_steps

    @property
    def minibatches_per_epoch(self):
        return self.steps_per_update // self.minibatch_size

    def validate(self):
        """Catch the sizing mistake that cost a full training run."""
        problems = []
        if self.minibatch_size >= self.steps_per_update:
            problems.append(
                f"minibatch_size {self.minibatch_size} >= batch "
                f"{self.steps_per_update} — 'minibatching' would be "
                f"full-batch descent"
            )
        if self.steps_per_update % self.minibatch_size != 0:
            problems.append(
                f"batch {self.steps_per_update} is not divisible by "
                f"minibatch_size {self.minibatch_size} — ragged last minibatch"
            )
        return problems


# ===========================================================================
# The profiles
# ===========================================================================
# Ordered most-specific first; `select()` takes the first match.

PROFILES = [
    # -- NVIDIA ------------------------------------------------------------
    TuneProfile(
        name="rtx3050-8gb",
        backend="cuda", num_envs=32, rollout_steps=64, minibatch_size=512,
        tf32=False, torch_threads=8, measured=True,
        rationale=(
            "MEASURED on an RTX 3050 8GB + i7-9700K (8c/8t, no SMT), the "
            "machine this repo's Stage 3 runs were developed on. Sustains "
            "1,050-1,170 steps/s. The bottleneck is MuJoCo stepping on the "
            "CPU, not the GPU -- the 3050 is largely idle during rollout "
            "collection, which is why raising num_envs further does not "
            "help on this box. tf32 off: measured to put a 0.0014 floor "
            "under approx_kl on a first minibatch where the exact answer "
            "is 0."
        ),
    ),
    TuneProfile(
        name="cuda-4gb",
        backend="cuda", num_envs=16, rollout_steps=64, minibatch_size=256,
        tf32=False, torch_threads=None, measured=False,
        rationale=(
            "Small-VRAM cards (GTX 1650, RTX 3050 4GB laptop, T500). Halved "
            "batch so the 2102-wide history tensors fit with headroom for "
            "the critic. Not benchmarked."
        ),
    ),
    TuneProfile(
        name="cuda-8gb",
        backend="cuda", num_envs=32, rollout_steps=64, minibatch_size=512,
        tf32=False, torch_threads=None, measured=False,
        rationale=(
            "Generic 8 GB CUDA card. Same shape as the measured rtx3050-8gb "
            "profile, without the CPU-specific thread pin."
        ),
    ),
    TuneProfile(
        name="cuda-16gb",
        backend="cuda", num_envs=64, rollout_steps=64, minibatch_size=1024,
        tf32=False, torch_threads=None, measured=False,
        rationale=(
            "RTX 4080/3090/A4000 class. Doubled envs and minibatch. Whether "
            "this actually pays depends on core count -- rollout collection "
            "is CPU-bound, so on a 8-core host this will NOT be 2x faster "
            "than the 8 GB profile. Benchmark before trusting it."
        ),
    ),
    TuneProfile(
        name="cuda-24gb+",
        backend="cuda", num_envs=128, rollout_steps=64, minibatch_size=2048,
        tf32=False, torch_threads=None, measured=False,
        rationale=(
            "RTX 4090 / A100 class, assumed paired with a many-core host. "
            "128 sequential MuJoCo envs only makes sense above ~24 physical "
            "cores; select() downgrades this automatically on a small CPU."
        ),
    ),

    # -- Intel Arc (XPU) ---------------------------------------------------
    # None of these are measured. They exist so an Arc contributor gets a
    # working GPU path instead of a silent CPU fallback, which is the state
    # the repo was in before.
    TuneProfile(
        name="arc-8gb",
        backend="xpu", num_envs=32, rollout_steps=64, minibatch_size=512,
        tf32=False, torch_threads=None, measured=False,
        rationale=(
            "Intel Arc A750 / A580 / B570 class, 8 GB. UNBENCHMARKED. "
            "Mirrors the measured 8 GB CUDA shape because the model is tiny "
            "(a few MB of weights) and the bottleneck is CPU-side MuJoCo "
            "either way. tf32 is a CUDA concept and is ignored on XPU."
        ),
    ),
    TuneProfile(
        name="arc-16gb",
        backend="xpu", num_envs=64, rollout_steps=64, minibatch_size=1024,
        tf32=False, torch_threads=None, measured=False,
        rationale=(
            "Intel Arc A770 / B580 16 GB. UNBENCHMARKED. Same caveat as "
            "cuda-16gb: the win is CPU-bound, so measure before believing."
        ),
    ),

    # -- Apple -------------------------------------------------------------
    TuneProfile(
        name="apple-mps",
        backend="mps", num_envs=16, rollout_steps=64, minibatch_size=256,
        tf32=False, torch_threads=None, measured=False,
        rationale=(
            "Apple Silicon unified memory. Conservative: MPS kernel launch "
            "overhead for tensors this small is often worse than CPU, so "
            "benchmark against --device cpu before assuming MPS is faster."
        ),
    ),

    # -- CPU ---------------------------------------------------------------
    TuneProfile(
        name="cpu-manycore",
        backend="cpu", num_envs=16, rollout_steps=64, minibatch_size=256,
        tf32=False, torch_threads=None, measured=False,
        rationale=(
            "16+ logical cores with no usable GPU. Still CPU-bound on MuJoCo "
            "stepping; the extra envs buy batch diversity, not speed."
        ),
    ),
    TuneProfile(
        name="cpu-default",
        backend="cpu", num_envs=8, rollout_steps=64, minibatch_size=128,
        tf32=False, torch_threads=None, measured=False,
        rationale=(
            "The repo's original CPU defaults. Small envs and minibatches "
            "because MuJoCo stepping dominates entirely and every extra env "
            "is pure serial cost."
        ),
    ),
]

BY_NAME = {p.name: p for p in PROFILES}


# ===========================================================================
# Selection
# ===========================================================================
def select(backend, vram_gb=None, gpu_name=None, logical_cores=None):
    """
    Pick a profile for this machine.

    Matching is on backend first, then VRAM, with a specific override for the
    one card that has actually been measured. Returns a TuneProfile; never
    returns None -- an unrecognised machine gets the conservative profile for
    its backend rather than an error, because a slow correct run beats a
    crash at hour three.
    """
    name = (gpu_name or "").lower()

    if backend == "cuda":
        # The measured profile wins for the exact card it was measured on.
        if "3050" in name and (vram_gb or 0) >= 6:
            chosen = BY_NAME["rtx3050-8gb"]
        elif vram_gb is None:
            chosen = BY_NAME["cuda-8gb"]
        elif vram_gb < 6:
            chosen = BY_NAME["cuda-4gb"]
        elif vram_gb < 12:
            chosen = BY_NAME["cuda-8gb"]
        elif vram_gb < 20:
            chosen = BY_NAME["cuda-16gb"]
        else:
            chosen = BY_NAME["cuda-24gb+"]

    elif backend == "xpu":
        chosen = (BY_NAME["arc-16gb"] if (vram_gb or 0) >= 12
                  else BY_NAME["arc-8gb"])

    elif backend == "mps":
        chosen = BY_NAME["apple-mps"]

    else:
        chosen = (BY_NAME["cpu-manycore"] if (logical_cores or 0) >= 16
                  else BY_NAME["cpu-default"])

    return _fit_to_cpu(chosen, logical_cores)


def _fit_to_cpu(profile, logical_cores):
    """
    Rollout collection steps MuJoCo sequentially in Python, so num_envs is
    bounded by the CPU regardless of how large the GPU is. A 4090 in a 4-core
    box cannot use 128 envs; it just makes each update take longer.

    Cap at 4x logical cores and re-derive minibatch_size to keep 4 minibatches
    per epoch.
    """
    if not logical_cores:
        return profile
    cap = max(8, 4 * logical_cores)
    if profile.num_envs <= cap:
        return profile

    num_envs = cap
    batch = num_envs * profile.rollout_steps
    minibatch = max(64, batch // 4)
    return TuneProfile(
        name=f"{profile.name}-cpucapped",
        backend=profile.backend, num_envs=num_envs,
        rollout_steps=profile.rollout_steps, minibatch_size=minibatch,
        tf32=profile.tf32, torch_threads=profile.torch_threads,
        measured=False,
        rationale=(
            f"{profile.rationale}\n\nDOWNSIZED from {profile.name} "
            f"(num_envs {profile.num_envs} -> {num_envs}): this host has "
            f"only {logical_cores} logical cores, and rollout collection is "
            f"sequential MuJoCo stepping, so the original env count would "
            f"add latency without adding throughput."
        ),
    )


def select_for_profile(hw):
    """Pick a profile from a `scripts/hw_profile.py` dict."""
    gpus = hw.get("gpus") or []
    gpu = gpus[0] if gpus else {}
    return select(
        backend=hw.get("backend", "cpu"),
        vram_gb=gpu.get("vram_gb"),
        gpu_name=gpu.get("name"),
        logical_cores=(hw.get("cpu") or {}).get("cores_logical"),
    )


def describe(profile):
    lines = [
        f"profile          {profile.name}"
        f"{'  (MEASURED)' if profile.measured else '  (estimated — unbenchmarked)'}",
        f"  backend        {profile.backend}",
        f"  num_envs       {profile.num_envs}",
        f"  rollout_steps  {profile.rollout_steps}",
        f"  minibatch_size {profile.minibatch_size}"
        f"   ({profile.minibatches_per_epoch} minibatches/epoch of "
        f"{profile.steps_per_update})",
        f"  tf32           {profile.tf32}",
    ]
    if profile.torch_threads:
        lines.append(f"  torch_threads  {profile.torch_threads}")
    lines.append("")
    for para in profile.rationale.split("\n\n"):
        lines.append(f"  {para}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import json
    import sys
    from pathlib import Path

    ap = argparse.ArgumentParser(description="show PPO tuning profiles")
    ap.add_argument("--all", action="store_true", help="list every profile")
    ap.add_argument("--for-host", help="select using a saved hardware profile")
    args = ap.parse_args()

    bad = 0
    for p in PROFILES:
        for problem in p.validate():
            print(f"INVALID {p.name}: {problem}")
            bad += 1
    if bad:
        sys.exit(1)

    if args.all:
        for p in PROFILES:
            print(describe(p))
            print("-" * 74)
        sys.exit(0)

    if args.for_host:
        path = (Path(__file__).resolve().parent.parent / "hardware"
                / "profiles" / f"{args.for_host}.json")
        hw = json.loads(path.read_text())
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                               / "scripts"))
        import hw_profile
        hw = hw_profile.build_profile()

    print(describe(select_for_profile(hw)))
