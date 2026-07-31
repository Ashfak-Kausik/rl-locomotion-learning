"""
Render the Go2 walking to an MP4, offscreen — no GUI window required.

Exists because the live GLFW viewer (`watch_walk.py`, `06_run_policy.py`, ...)
has proven unreliable on at least one dev machine: the window closes early
for reasons outside this repo (not an idle-blanking/DPMS cause — checked),
and every `mujoco.viewer.launch_passive` script segfaults on teardown even
after a fully successful run (see docs/NATIVE-SETUP.md). Offscreen rendering
via `mujoco.Renderer` + `imageio` sidesteps both: no window, no viewer
teardown, deterministic frame-by-frame output you can actually attach
somewhere.

Uses the same heading-hold fix as watch_walk.py (see
EXPERIMENT_FINDINGS.md Experiment 4) by default — pass --no-heading-hold to
reproduce the original open-loop yaw drift instead.

    python make_video.py out.mp4
    python make_video.py out.mp4 --scene go2_gauntlet --cmd-vx 0.5 --seconds 25
    python make_video.py out.mp4 --scene go2_obstacles_hard --gait bound
    GO2_POLICY_DIR=../../policies/stage3-multigait_v2 python make_video.py out.mp4
"""

import argparse
import os
import sys
from collections import deque

import imageio.v2 as iio
import mujoco
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paths import SCENES_DIR, POLICY_DIR, require_policy  # noqa: E402
import harness as H                                        # noqa: E402

K_HEADING, MAX_YAW = 1.5, 0.6


def yaw_of(quat):
    w, x, y, z = quat
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def render(out_path, scene, cmd_vx, gait, seconds, heading_hold,
          width=960, height=540, fps=50):
    require_policy()
    body_net = torch.jit.load(f"{POLICY_DIR}/body_latest.jit"); body_net.eval()
    adapt_net = torch.jit.load(f"{POLICY_DIR}/adaptation_module_latest.jit")
    adapt_net.eval()

    model = mujoco.MjModel.from_xml_path(
        os.path.join(SCENES_DIR, scene + ".xml"))
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    data.qpos[7:19] = H.DEFAULT_JOINT_POS
    data.qpos[2] = 0.30
    mujoco.mj_forward(model, data)

    gait_params = H.GAIT_PRESETS[gait]
    commands_vec = np.array([
        cmd_vx, 0.0, 0.0, 0.0, H.STEP_FREQUENCY,
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
    n_steps = int(seconds / dt)
    every = max(1, int(round(1.0 / (fps * dt))))

    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.azimuth, cam.elevation = 3.0, 130, -14

    frames = []
    for step in range(n_steps):
        if step % H.DECIMATION == 0:
            if heading_hold:
                err_rad = float(yaw_of(data.qpos[3:7]))
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

        if step % every == 0:
            cam.lookat[:] = data.qpos[:3]
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())

    renderer.close()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    iio.mimsave(out_path, frames, fps=fps, quality=8)

    return dict(
        out_path=out_path, n_frames=len(frames),
        travelled_x=float(data.qpos[0]), travelled_y=float(data.qpos[1]),
        final_height=float(data.qpos[2]),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="output .mp4 path")
    ap.add_argument("--scene", default="go2_flat",
                    help="scene name without .xml, e.g. go2_gauntlet")
    ap.add_argument("--cmd-vx", type=float, default=0.5)
    ap.add_argument("--gait", default="trot", choices=list(H.GAIT_PRESETS))
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--no-heading-hold", action="store_true",
                    help="disable the F4 fix, to see the original drift")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fps", type=int, default=50)
    args = ap.parse_args()

    result = render(args.out, args.scene, args.cmd_vx, args.gait, args.seconds,
                    heading_hold=not args.no_heading_hold,
                    width=args.width, height=args.height, fps=args.fps)

    print(f"{result['out_path']}  {result['n_frames']} frames  "
          f"travelled x={result['travelled_x']:.2f}m "
          f"y={result['travelled_y']:+.2f}m "
          f"height={result['final_height']:.3f}m")


if __name__ == "__main__":
    main()
