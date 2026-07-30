"""
Export a trained checkpoint to TorchScript in the Stage 2 contract.

This script is the whole point of Stage 3's interface discipline. It writes
exactly the two files Stage 2 already knows how to load:

    body_latest.jit                 (1, 2102) -> (1, 12)
    adaptation_module_latest.jit    (1, 2100) -> (1, 2)

which means a policy trained here is immediately evaluable by
`stage2-go2-mujoco-inference/experiments/harness.py` — same scenes, same
metrics, same figures — and directly comparable against the walk-these-ways
baseline numbers in EXPERIMENT_FINDINGS.md.

Nothing is written until the contract has been verified three ways:
  1. shape check on the live nn.Modules
  2. shape check after TorchScript tracing
  3. numerical equivalence between the eager and traced graphs

Usage
-----
    python export.py --checkpoint runs/my_run/checkpoint_final.pt
    python export.py --checkpoint ... --out-dir ../policies/my-policy
    python export.py --random --out-dir ../policies/random-baseline
        (untrained but contract-valid — useful for testing the pipeline
         end to end without any training, and as a control condition)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from networks import (  # noqa: E402
    ACTION_DIM,
    BODY_INPUT_DIM,
    HISTORY_DIM,
    LATENT_DIM,
    AdaptationModule,
    Body,
    Go2Agent,
    assert_contract,
)

BODY_FILENAME = "body_latest.jit"
ADAPT_FILENAME = "adaptation_module_latest.jit"


def load_agent(checkpoint_path, device="cpu"):
    agent = Go2Agent().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()
    return agent, ckpt


def verify_traced(traced_adapt, traced_body, eager_adapt, eager_body,
                  tolerance=1e-5):
    """
    Confirm the traced graphs match the eager modules numerically.

    Tracing silently bakes in control flow that depended on tensor values. The
    networks here are pure MLPs so that cannot bite, but verifying costs
    microseconds and the failure it catches is otherwise invisible until the
    robot behaves oddly.
    """
    torch.manual_seed(0)
    for batch in (1, 4):
        history = torch.randn(batch, HISTORY_DIM)
        with torch.no_grad():
            lat_e = eager_adapt(history)
            lat_t = traced_adapt(history)
            if not torch.allclose(lat_e, lat_t, atol=tolerance):
                raise ValueError(
                    f"traced adaptation module diverges from eager: "
                    f"max |diff| = {(lat_e - lat_t).abs().max().item():.3e}"
                )

            body_in = torch.cat([history, lat_e], dim=1)
            act_e = eager_body(body_in)
            act_t = traced_body(body_in)
            if not torch.allclose(act_e, act_t, atol=tolerance):
                raise ValueError(
                    f"traced body diverges from eager: "
                    f"max |diff| = {(act_e - act_t).abs().max().item():.3e}"
                )
    return True


def verify_roundtrip(out_dir):
    """
    Reload the written files exactly as Stage 2 does, and push a tensor
    through the full pipeline. This is the real test: if this passes,
    harness.py will work.
    """
    body = torch.jit.load(str(Path(out_dir) / BODY_FILENAME))
    adapt = torch.jit.load(str(Path(out_dir) / ADAPT_FILENAME))
    body.eval()
    adapt.eval()

    with torch.no_grad():
        history = torch.zeros(1, HISTORY_DIM, dtype=torch.float32)
        latent = adapt(history)
        assert tuple(latent.shape) == (1, LATENT_DIM), latent.shape

        body_input = torch.cat([history, latent], dim=1)
        assert tuple(body_input.shape) == (1, BODY_INPUT_DIM), body_input.shape

        action = body(body_input).numpy().flatten()
        assert action.shape == (ACTION_DIM,), action.shape
        assert bool(torch.isfinite(torch.tensor(action)).all())

    return action


def export(adaptation_module, body, out_dir, metadata=None):
    """Trace, verify, then write. Never writes an unverified graph."""
    out_dir = Path(out_dir)
    adaptation_module = adaptation_module.cpu().eval()
    body = body.cpu().eval()

    # 1. eager contract
    assert_contract(adaptation_module, body)

    # 2. trace
    with torch.no_grad():
        example_history = torch.zeros(1, HISTORY_DIM)
        traced_adapt = torch.jit.trace(adaptation_module, example_history)

        example_body_in = torch.zeros(1, BODY_INPUT_DIM)
        traced_body = torch.jit.trace(body, example_body_in)

    # 3. eager vs traced
    verify_traced(traced_adapt, traced_body, adaptation_module, body)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced_body, str(out_dir / BODY_FILENAME))
    torch.jit.save(traced_adapt, str(out_dir / ADAPT_FILENAME))

    # 4. reload and run, exactly as Stage 2 will
    action = verify_roundtrip(out_dir)

    meta = {
        "contract": {
            "obs_dim": 70,
            "history_len": 30,
            "history_dim": HISTORY_DIM,
            "latent_dim": LATENT_DIM,
            "body_input_dim": BODY_INPUT_DIM,
            "action_dim": ACTION_DIM,
        },
        "files": {"body": BODY_FILENAME, "adaptation_module": ADAPT_FILENAME},
        "verified": {
            "eager_shapes": True,
            "traced_shapes": True,
            "eager_vs_traced_numerics": True,
            "reload_roundtrip": True,
        },
        "sample_action_on_zero_input": [round(float(a), 6) for a in action],
    }
    if metadata:
        meta.update(metadata)
    (out_dir / "export_metadata.json").write_text(json.dumps(meta, indent=2))

    return out_dir


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", help="path to checkpoint_*.pt")
    group.add_argument("--random", action="store_true",
                       help="export freshly initialised (untrained) networks — "
                            "contract-valid, useful as a pipeline test and as "
                            "a control condition")
    ap.add_argument("--out-dir", default=None,
                    help="default: <repo>/policies/stage3-<run name>")
    ap.add_argument("--seed", type=int, default=0,
                    help="only used with --random")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    if args.random:
        torch.manual_seed(args.seed)
        adapt, body = AdaptationModule(), Body()
        metadata = {"source": "randomly initialised", "seed": args.seed,
                    "trained": False}
        default_out = repo_root / "policies" / "stage3-random"
    else:
        agent, ckpt = load_agent(args.checkpoint)
        adapt, body = agent.adaptation_module, agent.body
        cfg = ckpt.get("config", {})
        metadata = {
            "source": str(args.checkpoint),
            "trained": True,
            "global_step": ckpt.get("global_step"),
            "update": ckpt.get("update"),
            "run_name": cfg.get("run_name"),
            "gait": cfg.get("gait"),
            "domain_rand": cfg.get("domain_rand"),
        }
        run = cfg.get("run_name", "run")
        default_out = repo_root / "policies" / f"stage3-{run}"

    out_dir = Path(args.out_dir) if args.out_dir else default_out
    out_dir = export(adapt, body, out_dir, metadata)

    print(f"exported to {out_dir}")
    print(f"  {BODY_FILENAME}              (1, {BODY_INPUT_DIM}) -> (1, {ACTION_DIM})")
    print(f"  {ADAPT_FILENAME}  (1, {HISTORY_DIM}) -> (1, {LATENT_DIM})")
    print("  export_metadata.json")
    print("\ncontract verified: eager shapes, traced shapes, "
          "eager-vs-traced numerics, reload round-trip")
    print("\nEvaluate it with Stage 2's harness, unmodified:")
    print(f"  export GO2_POLICY_DIR={out_dir}")
    print("  python stage3-go2-training/evaluate.py")
    print("  python stage2-go2-mujoco-inference/experiments/exp1_velocity_sweep.py")


if __name__ == "__main__":
    main()
