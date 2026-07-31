"""
Experiment 4 — heading hold, and the frame in which velocity is measured.

WHY THIS EXISTS
---------------
Experiment 1 reported that the baseline tracks only 37-55% of commanded
forward velocity. Reproducing it turned up something the metric hid.

`harness.run_trial` logs `data.qvel[0]`, which is WORLD-frame x velocity, and
compares it against `lin_vel_x`, which `build_obs` hands the policy as a
BODY-frame command. Those are the same number only while the robot faces +x.

It does not. `ang_vel_yaw` is commanded at a constant 0.0 and nothing closes
the loop on heading, so any yaw bias integrates freely. At cmd 0.5 the robot
turns roughly -70 degrees over a 30 s trial and walks a slow arc. Body-frame
speed stays flat at ~0.296 m/s the whole time while the world-frame projection
decays from 0.296 to 0.135 -- pure cos(yaw), not deceleration.

This experiment measures two things:

  1. the size of that artefact, per commanded speed (both frames, side by side)
  2. whether closing a heading loop removes it

The heading controller is the obvious one, and is what a real deployment does
with a joystick or a waypoint follower:

    ang_vel_yaw = clip(-K_HEADING * yaw_error, -MAX_YAW, +MAX_YAW)

It needs no retraining: `ang_vel_yaw` is already an input the policy was
trained on, it was simply never used.

Experiments 1-3 are deliberately NOT modified -- their CSVs stay reproducible
from their own source. This is a separate, additive study.

    python exp4_heading_hold.py
"""

import csv
import os
import sys
from collections import deque

import numpy as np
import torch
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paths import SCENES_DIR, require_policy, POLICY_DIR  # noqa: E402
import harness as H                                       # noqa: E402

# ==== CONFIGURATION =========================================================
COMMANDS = [0.25, 0.50, 0.75, 1.00]
SEEDS = [0, 1, 2]
GAIT = "trot"
SETTLE_S = 3.0
MEASURE_S = 30.0

# Heading controller. K is deliberately gentle: the policy was trained on
# yaw-rate commands in roughly +-1 rad/s, and saturating that input degrades
# the gait. MAX_YAW keeps it well inside the trained range.
K_HEADING = 1.5     # rad/s of commanded yaw rate per rad of heading error
MAX_YAW = 0.6       # rad/s

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT_CSV = os.path.join(RESULTS_DIR, "exp4_heading_hold.csv")


def run(cmd_vx, seed, heading_hold):
    """One trial. Mirrors harness.run_trial, plus optional heading feedback."""
    model = mujoco.MjModel.from_xml_path(os.path.join(SCENES_DIR, "go2_flat.xml"))
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    data.qpos[7:19] = H.DEFAULT_JOINT_POS
    data.qpos[2] = 0.30

    # Identical seeding to harness.run_trial, so trials are comparable.
    rng = np.random.default_rng(seed)
    data.qpos[7:19] += rng.uniform(-0.02, 0.02, size=12)
    data.qpos[2] += rng.uniform(-0.01, 0.01)
    yaw0 = rng.uniform(-0.05, 0.05)
    data.qpos[3] = np.cos(yaw0 / 2.0)
    data.qpos[4] = 0.0
    data.qpos[5] = 0.0
    data.qpos[6] = np.sin(yaw0 / 2.0)
    data.qvel[6:18] += rng.uniform(-0.05, 0.05, size=12)
    mujoco.mj_forward(model, data)

    commands_vec = np.array([
        cmd_vx, 0.0, 0.0, 0.0, H.STEP_FREQUENCY,
        *H.GAIT_PRESETS[GAIT], 0.5, 0.06, 0.0, 0.0, 0.0, 0.0, 0.0,
    ])
    gait_params = H.GAIT_PRESETS[GAIT]

    obs_history = deque(
        [np.zeros(H.OBS_DIM, dtype=np.float32) for _ in range(H.HISTORY_LEN)],
        maxlen=H.HISTORY_LEN,
    )
    last_action = np.zeros(12)
    prev_action = np.zeros(12)
    joint_targets = H.DEFAULT_JOINT_POS.copy()
    gait_phase_t = 0.0

    action_scale_per_joint = np.full(12, H.ACTION_SCALE)
    for hip_idx in (0, 3, 6, 9):
        action_scale_per_joint[hip_idx] *= H.HIP_SCALE_REDUCTION

    dt = model.opt.timestep
    n_steps = int((SETTLE_S + MEASURE_S) / dt)
    measure_start = int(SETTLE_S / dt)

    vx_w, vx_b, yaws, ys = [], [], [], []
    fell = False

    for step in range(n_steps):
        if step % H.DECIMATION == 0:
            if heading_hold:
                err_rad = np.radians(H.yaw_deg(data.qpos[3:7]))   # target = 0
                commands_vec[2] = float(
                    np.clip(-K_HEADING * err_rad, -MAX_YAW, MAX_YAW))
            gait_phase_t = (gait_phase_t
                            + H.STEP_FREQUENCY * dt * H.DECIMATION) % 1.0
            obs = H.build_obs(data, commands_vec, gait_params,
                              prev_action, last_action, gait_phase_t)
            obs_history.append(obs)
            hist = torch.tensor(
                np.concatenate(obs_history).reshape(1, -1), dtype=torch.float32)
            with torch.no_grad():
                latent = ADAPT_NET(hist)
                action = BODY_NET(torch.cat([hist, latent], dim=1)).numpy().flatten()
            prev_action = last_action.copy()
            last_action = action.copy()
            joint_targets = H.DEFAULT_JOINT_POS + action * action_scale_per_joint

        data.ctrl[:] = (H.KP * (joint_targets - data.qpos[7:19])
                        - H.KD * data.qvel[6:18])
        mujoco.mj_step(model, data)

        if (data.qpos[2] < H.FALL_HEIGHT_THRESHOLD
                or H.tilt_angle_deg(data.qpos[3:7]) > H.FALL_TILT_THRESHOLD_DEG):
            fell = True
            break

        if step >= measure_start:
            v_world = data.qvel[0:3].copy()
            v_body = H.quat_rotate_inverse(data.qpos[3:7].copy(), v_world)
            vx_w.append(v_world[0])
            vx_b.append(v_body[0])
            yaws.append(H.yaw_deg(data.qpos[3:7]))
            ys.append(data.qpos[1])

    if fell:
        # null, not 0 — consistent with the rest of the repo.
        return dict(cmd_vx=cmd_vx, seed=seed, heading_hold=heading_hold,
                    fell=True, mean_vx_world=None, mean_vx_body=None,
                    yaw_drift_deg=None, lateral_offset_m=None)

    return dict(
        cmd_vx=cmd_vx, seed=seed, heading_hold=heading_hold, fell=False,
        mean_vx_world=round(float(np.mean(vx_w)), 4),
        mean_vx_body=round(float(np.mean(vx_b)), 4),
        yaw_drift_deg=round(float(yaws[-1] - yaws[0]), 2),
        lateral_offset_m=round(float(abs(ys[-1])), 3),
    )


if __name__ == "__main__":
    require_policy()
    BODY_NET = torch.jit.load(f"{POLICY_DIR}/body_latest.jit")
    BODY_NET.eval()
    ADAPT_NET = torch.jit.load(f"{POLICY_DIR}/adaptation_module_latest.jit")
    ADAPT_NET.eval()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    total = len(COMMANDS) * len(SEEDS) * 2
    n = 0
    for cmd in COMMANDS:
        for hold in (False, True):
            for seed in SEEDS:
                n += 1
                print(f"[{n:2d}/{total}] cmd={cmd:.2f} "
                      f"hold={'on ' if hold else 'off'} seed={seed}", flush=True)
                rows.append(run(cmd, seed, hold))

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {OUT_CSV}\n")
    hdr = (f"{'cmd':>5} | {'hold':>4} | {'world vx':>9} | {'body vx':>9} | "
           f"{'yaw drift':>10} | {'|y| end':>8}")
    print(hdr)
    print("-" * len(hdr))
    for cmd in COMMANDS:
        for hold in (False, True):
            sel = [r for r in rows
                   if r["cmd_vx"] == cmd and r["heading_hold"] is hold
                   and not r["fell"]]
            if not sel:
                print(f"{cmd:5.2f} | {'on' if hold else 'off':>4} | all trials fell")
                continue
            print(f"{cmd:5.2f} | {'on' if hold else 'off':>4} | "
                  f"{np.mean([r['mean_vx_world'] for r in sel]):9.3f} | "
                  f"{np.mean([r['mean_vx_body'] for r in sel]):9.3f} | "
                  f"{np.mean([abs(r['yaw_drift_deg']) for r in sel]):9.1f}d | "
                  f"{np.mean([r['lateral_offset_m'] for r in sel]):7.2f}m")
        print()
