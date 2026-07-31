"""
Live viewer: the Go2 walking straight, with heading hold applied.

06_run_policy.py and 07_run_policy_interactive.py command a constant
ang_vel_yaw of 0.0 with no heading feedback, so the robot slowly curves (see
exp4_heading_hold.py and EXPERIMENT_FINDINGS.md F4.1). This script is a
`--watch` tool, not a curriculum lesson (01-06 are deliberately left alone
per CLAUDE.md), for actually looking at the policy without the drift.

    python watch_walk.py                          # trot, flat, 0.5 m/s
    python watch_walk.py --scene go2_slope_10 --cmd-vx 0.6
    python watch_walk.py --gait pace --no-heading-hold   # reproduce the drift
"""

import argparse
import os
import sys
import time
from collections import deque

import numpy as np
import torch
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paths import SCENES_DIR, require_policy, POLICY_DIR  # noqa: E402
import harness as H                                        # noqa: E402

K_HEADING = 1.5    # rad/s of commanded yaw rate per rad of heading error
MAX_YAW = 0.6      # rad/s — stays inside the policy's trained command range


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="go2_flat")
    ap.add_argument("--cmd-vx", type=float, default=0.5)
    ap.add_argument("--gait", default="trot", choices=list(H.GAIT_PRESETS))
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--no-heading-hold", action="store_true",
                    help="disable the fix, to see the original drift")
    args = ap.parse_args()

    require_policy()
    body_net = torch.jit.load(f"{POLICY_DIR}/body_latest.jit"); body_net.eval()
    adapt_net = torch.jit.load(f"{POLICY_DIR}/adaptation_module_latest.jit")
    adapt_net.eval()

    scene_path = os.path.join(SCENES_DIR, args.scene + ".xml")
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    data.qpos[7:19] = H.DEFAULT_JOINT_POS
    data.qpos[2] = 0.30
    mujoco.mj_forward(model, data)

    gait_params = H.GAIT_PRESETS[args.gait]
    commands_vec = np.array([
        args.cmd_vx, 0.0, 0.0, 0.0, H.STEP_FREQUENCY,
        *gait_params, 0.5, 0.06, 0.0, 0.0, 0.0, 0.0, 0.0,
    ])

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
    hold = not args.no_heading_hold
    print(f"scene={args.scene} cmd_vx={args.cmd_vx} gait={args.gait} "
          f"heading_hold={hold}")
    print("Ctrl+C in this terminal, or close the window, to stop.")

    step = 0
    start = time.time()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and time.time() - start < args.seconds:
            step_start = time.time()
            if step % H.DECIMATION == 0:
                if hold:
                    err_rad = np.radians(H.yaw_deg(data.qpos[3:7]))
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
                    latent = adapt_net(hist)
                    action = body_net(
                        torch.cat([hist, latent], dim=1)).numpy().flatten()
                prev_action = last_action.copy()
                last_action = action.copy()
                joint_targets = (H.DEFAULT_JOINT_POS
                                 + action * action_scale_per_joint)

            data.ctrl[:] = (H.KP * (joint_targets - data.qpos[7:19])
                            - H.KD * data.qvel[6:18])
            mujoco.mj_step(model, data)
            step += 1
            viewer.sync()

            if step % 500 == 0:
                v_body = H.quat_rotate_inverse(
                    data.qpos[3:7].copy(), data.qvel[0:3].copy())
                print(f"  t={data.time:6.2f}s  pos=({data.qpos[0]:+.2f},"
                      f"{data.qpos[1]:+.2f})  body_vx={v_body[0]:+.3f}  "
                      f"yaw={H.yaw_deg(data.qpos[3:7]):+6.1f}deg  "
                      f"height={data.qpos[2]:.3f}m")

            remaining = dt - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)

    print(f"Done. Travelled x={data.qpos[0]:+.2f}m y={data.qpos[1]:+.2f}m "
          f"over {data.time:.1f}s of sim time.")


if __name__ == "__main__":
    main()
