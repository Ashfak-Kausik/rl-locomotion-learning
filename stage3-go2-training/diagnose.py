"""
Answer two questions about a training checkpoint, with numbers:

    1. Does this policy actually WALK?
    2. If the run went wrong, WHICH reward term did it?

Both were learned the hard way. A training log tells you the return, and the
return alone is close to useless for diagnosis:

  * multigait_v4 reached a stable return of ~-19 and looked converged. It was
    standing perfectly still. Return was stable because `alive` (0.500/step)
    plus `tracking_ang_vel` (maximised by not turning) dominated
    `tracking_lin_vel` (0.029/step), so standing WAS the optimum. Nothing in
    the log said so.

  * multigait_v3 hit -6,454 and the log just showed a big negative number.
    The per-term breakdown showed `joint_torque` at -217.98/step -- a single
    term, from actions of magnitude ~59 against a policy whose real output
    range is about +-1. That pinned it to the action clip in one run.

So: roll the policy out, measure the distance it covers in the world, and
print `info["reward_terms"]` summed per term. A policy that is not moving and
a policy that is being strangled by one penalty look identical in the return
column and completely different here.

Runs on CPU by default so it can be used while a GPU training job is still
going.

Usage
-----
    python diagnose.py --run multigait_v5                 # best checkpoint
    python diagnose.py --run multigait_v5 --checkpoint latest
    python diagnose.py --run multigait_v5 --cmd-vx 1.0 --seconds 20
    python diagnose.py --run multigait_v5 --episodes 5    # average over seeds
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE3 = REPO_ROOT / "stage3-go2-training"
if str(STAGE3) not in sys.path:
    sys.path.insert(0, str(STAGE3))

from config import Config          # noqa: E402
from env.go2_env import Go2Env     # noqa: E402
from networks import Go2Agent      # noqa: E402

# A policy that covers less than this in the whole episode is standing still,
# whatever its return says. 0.5 m over 10 s is 0.05 m/s -- drift, not gait.
STANDING_STILL_M = 0.5


def rollout(agent, cfg, seed, command, max_steps, gait, scene):
    """One episode. Returns (per-term totals, summary dict)."""
    env = Go2Env(cfg, seed=seed, scene=scene)
    obs, priv = env.reset(command=command)

    start_xy = env.data.qpos[0:2].copy()
    totals, vxs, heights = {}, [], []
    ep_return, n, fell = 0.0, 0, False

    for _ in range(max_steps):
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            p = torch.as_tensor(priv, dtype=torch.float32).unsqueeze(0)
            # Deterministic: we are measuring the learned mean, not the
            # exploration noise around it.
            action, _, _, _ = agent.act(o, p, deterministic=True)

        obs, priv, reward, term, trunc, info = env.step(
            action.numpy().squeeze())

        for k, v in info["reward_terms"].items():
            totals[k] = totals.get(k, 0.0) + v

        # Body-frame forward speed, not qvel[0] -- see F4.1. A robot that has
        # turned 90 degrees has a world-frame vx of ~0 while walking fine.
        from env.go2_env import quat_rotate_inverse
        v_body = quat_rotate_inverse(env.data.qpos[3:7].copy(),
                                     env.data.qvel[0:3].copy())
        vxs.append(float(v_body[0]))
        heights.append(float(env.data.qpos[2]))

        ep_return += reward
        n += 1
        if term or trunc:
            fell = bool(term)
            break

    end_xy = env.data.qpos[0:2].copy()
    return totals, {
        "steps": n,
        "seconds": n * env.dt_policy,
        "return": ep_return,
        "fell": fell,
        "distance_m": float(np.linalg.norm(end_xy - start_xy)),
        "displacement_x": float(end_xy[0] - start_xy[0]),
        "displacement_y": float(end_xy[1] - start_xy[1]),
        "mean_vx_body": float(np.mean(vxs)) if vxs else 0.0,
        "mean_height": float(np.mean(heights)) if heights else 0.0,
        "gait": gait,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run name under runs/")
    ap.add_argument("--checkpoint", default="best",
                    choices=["best", "latest", "final"])
    ap.add_argument("--cmd-vx", type=float, default=0.5)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--gait", default="trot")
    ap.add_argument("--scene", default="go2_flat.xml")
    ap.add_argument("--device", default="cpu",
                    help="cpu by default so this is safe to run during "
                         "a GPU training job")
    args = ap.parse_args()

    ckpt_path = STAGE3 / "runs" / args.run / f"checkpoint_{args.checkpoint}.pt"
    if not ckpt_path.is_file():
        available = sorted(p.name for p in ckpt_path.parent.glob("*.pt"))
        sys.exit(f"no such checkpoint: {ckpt_path}\n"
                 f"available: {available or 'none'}")

    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    agent = Go2Agent().to(args.device)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()

    print(f"\ncheckpoint  {ckpt_path.relative_to(REPO_ROOT)}")
    print(f"  step               {ckpt.get('global_step', 0):,}")
    if "best_mean_return" in ckpt:
        print(f"  best_mean_return   {ckpt['best_mean_return']:.1f}")
    print(f"  command            vx={args.cmd_vx} gait={args.gait} "
          f"scene={args.scene}")

    # Domain randomisation and gait randomisation OFF: we want a clean,
    # repeatable measurement of the policy, not of the randomisation.
    cfg = Config(curriculum=False, domain_rand=False,
                 randomize_gait=False, gait=args.gait)
    max_steps = int(args.seconds / (1.0 / 50.0))

    agg_totals, summaries = {}, []
    for ep in range(args.episodes):
        totals, summary = rollout(agent, cfg, seed=ep,
                                  command=(args.cmd_vx, 0.0, 0.0),
                                  max_steps=max_steps, gait=args.gait,
                                  scene=args.scene)
        for k, v in totals.items():
            agg_totals[k] = agg_totals.get(k, 0.0) + v
        summaries.append(summary)

    total_steps = sum(s["steps"] for s in summaries)
    mean = lambda k: float(np.mean([s[k] for s in summaries]))  # noqa: E731

    print(f"\n{'BEHAVIOUR':-<62}")
    print(f"  episodes            {args.episodes}"
          f"  ({total_steps} steps, {mean('seconds'):.1f} s each)")
    print(f"  distance travelled  {mean('distance_m'):8.2f} m"
          f"   (x {mean('displacement_x'):+.2f}, "
          f"y {mean('displacement_y'):+.2f})")
    print(f"  mean vx (body)      {mean('mean_vx_body'):8.3f} m/s"
          f"   commanded {args.cmd_vx}")
    print(f"  mean body height    {mean('mean_height'):8.3f} m"
          f"   nominal 0.30")
    print(f"  episode return      {mean('return'):8.1f}")
    fell = sum(s["fell"] for s in summaries)
    print(f"  fell                {fell}/{args.episodes}")

    verdict = "WALKS"
    if fell == args.episodes:
        verdict = "FALLS — terminates every episode"
    elif mean("distance_m") < STANDING_STILL_M:
        verdict = (f"STANDS STILL — {mean('distance_m'):.2f} m in "
                   f"{mean('seconds'):.0f} s")
    elif mean("mean_vx_body") < 0.3 * args.cmd_vx:
        verdict = (f"CRAWLS — {mean('mean_vx_body'):.2f} m/s against a "
                   f"{args.cmd_vx} m/s command")
    print(f"\n  VERDICT: {verdict}")

    print(f"\n{'REWARD BREAKDOWN (per step, all episodes)':-<62}")
    print(f"  {'term':24} {'per-step':>11} {'share of |total|':>17}")
    print(f"  {'-' * 58}")
    abs_sum = sum(abs(v) for v in agg_totals.values()) or 1.0
    for k, v in sorted(agg_totals.items(), key=lambda kv: -abs(kv[1])):
        share = 100.0 * abs(v) / abs_sum
        bar = "#" * int(share / 2)
        print(f"  {k:24} {v / total_steps:11.4f} {share:15.1f}%  {bar}")
    print(f"  {'-' * 58}")
    print(f"  {'TOTAL':24} {sum(agg_totals.values()) / total_steps:11.4f}")

    # The single most useful line: is the objective term actually driving?
    obj = (agg_totals.get("tracking_lin_vel", 0.0)
           + agg_totals.get("forward_progress", 0.0))
    obj_share = 100.0 * abs(obj) / abs_sum
    print(f"\n  objective terms (tracking_lin_vel + forward_progress) "
          f"= {obj_share:.1f}% of reward mass")
    if obj_share < 15.0:
        print("  ^ TOO LOW. The policy is being paid mostly for something "
              "other than\n    moving at the commanded speed. Expect it to "
              "optimise that instead.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
