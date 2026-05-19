"""
Stage 2 — Scene viewer for paper screenshots.

Runs the pretrained walk-these-ways policy in the MuJoCo viewer on a
SELECTABLE scene (flat / slope / stairs). Runs INDEFINITELY until you
press Ctrl+C in the terminal (or close the viewer window), so you have
unlimited time to position the camera and use the viewer's Screenshot
button for the best shot.

Usage:
    python3 08_view_scene.py                       # flat (default)
    python3 08_view_scene.py go2_slope_10.xml
    python3 08_view_scene.py go2_stairs_08.xml
    python3 08_view_scene.py go2_slope_25.xml

Controls (same arrow-key scheme as script 07):
    Up/Down    : forward / backward velocity
    Left/Right : yaw left / right
    , / .      : strafe
    0          : stop
    8 / 9 / 7  : trot / pace / bound
    R          : reset robot

Screenshot: use the viewer's File > Screenshot button (top-left panel).
PNGs are saved to the current working directory by MuJoCo.

Quit: Ctrl+C in the terminal, or close the viewer window.
"""

import sys
import os
import time
import datetime
import numpy as np
import torch
import mujoco
import mujoco.viewer
import imageio.v2 as imageio
from collections import deque

SHOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "paper_figures", "sim_shots")
)
os.makedirs(SHOT_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# Scene selection
# ----------------------------------------------------------------------------
scene_name = sys.argv[1] if len(sys.argv) > 1 else "go2_flat.xml"
SCENE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "scenes", scene_name)
)
if not os.path.exists(SCENE_PATH):
    print(f"ERROR: scene not found: {SCENE_PATH}")
    print("Available scenes:")
    sd = os.path.join(os.path.dirname(__file__), "scenes")
    for f in sorted(os.listdir(sd)):
        if f.endswith(".xml"):
            print(f"  {f}")
    sys.exit(1)

# ----------------------------------------------------------------------------
# Constants (identical to harness / script 07)
# ----------------------------------------------------------------------------
POLICY_DIR = (
    "/home/user/projects/robot-dog-sim/walk-these-ways-go2/runs/"
    "gait-conditioned-agility/pretrain-go2/train/142238.667503/checkpoints"
)
DEFAULT_JOINT_POS = np.array([
     0.1, 0.8, -1.5,  -0.1, 0.8, -1.5,
     0.1, 1.0, -1.5,  -0.1, 1.0, -1.5,
])
OBS_SCALES = {
    "lin_vel": 2.0, "ang_vel": 0.25, "dof_pos": 1.0, "dof_vel": 0.05,
    "body_height_cmd": 2.0, "gait_phase_cmd": 1.0, "gait_freq_cmd": 1.0,
    "footswing_height_cmd": 0.15, "body_pitch_cmd": 0.3,
    "body_roll_cmd": 0.3, "stance_width_cmd": 1.0,
    "stance_length_cmd": 1.0, "aux_reward_cmd": 1.0,
}
COMMANDS_SCALE = np.array([
    OBS_SCALES["lin_vel"], OBS_SCALES["lin_vel"], OBS_SCALES["ang_vel"],
    OBS_SCALES["body_height_cmd"], OBS_SCALES["gait_freq_cmd"],
    OBS_SCALES["gait_phase_cmd"], OBS_SCALES["gait_phase_cmd"],
    OBS_SCALES["gait_phase_cmd"], OBS_SCALES["gait_phase_cmd"],
    OBS_SCALES["footswing_height_cmd"], OBS_SCALES["body_pitch_cmd"],
    OBS_SCALES["body_roll_cmd"], OBS_SCALES["stance_width_cmd"],
    OBS_SCALES["stance_length_cmd"], OBS_SCALES["aux_reward_cmd"],
])
GAIT_PRESETS = {"trot": (0.5, 0.0, 0.0),
                "pace": (0.0, 0.5, 0.0),
                "bound": (0.0, 0.0, 0.5)}
ACTION_SCALE = 0.25
HIP_SCALE_REDUCTION = 0.5
KP, KD = 25.0, 0.6
DECIMATION = 10
HISTORY_LEN = 30
OBS_DIM = 70
STEP_FREQUENCY = 2.0


class ControlState:
    def __init__(self):
        self.lin_vel_x = 0.0
        self.lin_vel_y = 0.0
        self.ang_vel_yaw = 0.0
        self.gait_name = "trot"
        self.reset_requested = False
        self.screenshot_requested = False

    @property
    def gait_params(self):
        return GAIT_PRESETS[self.gait_name]

    def command_vec(self):
        p, o, b = self.gait_params
        return np.array([
            self.lin_vel_x, self.lin_vel_y, self.ang_vel_yaw,
            0.0, STEP_FREQUENCY, p, o, b, 0.5, 0.06,
            0.0, 0.0, 0.0, 0.0, 0.0,
        ])


def quat_rotate_inverse(q, v):
    q_w = q[0]
    q_vec = q[1:]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (2.0 * q_w)
    c = q_vec * 2.0 * (q_vec @ v)
    return a - b + c


def build_obs(data, cstate, prev_a, last_a, phase_t):
    proj_g = quat_rotate_inverse(data.qpos[3:7], np.array([0.0, 0.0, -1.0]))
    cmd = cstate.command_vec() * COMMANDS_SCALE
    jp = (data.qpos[7:19] - DEFAULT_JOINT_POS) * OBS_SCALES["dof_pos"]
    jv = data.qvel[6:18] * OBS_SCALES["dof_vel"]
    p, o, b = cstate.gait_params
    fp = np.array([phase_t + p + o + b, phase_t + o,
                   phase_t + b, phase_t + p]) % 1.0
    clock = np.sin(2 * np.pi * fp)
    return np.concatenate([proj_g, cmd, jp, jv, last_a, prev_a, clock])


def reset_robot(model, data):
    k = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if k >= 0:
        mujoco.mj_resetDataKeyframe(model, data, k)
    data.qpos[7:19] = DEFAULT_JOINT_POS
    data.qpos[2] = 0.30
    mujoco.mj_forward(model, data)


def make_key_callback(cstate):
    DV, DW, VMAX, WMAX = 0.25, 0.25, 1.5, 1.5

    def cb(keycode):
        if keycode == 265:    # Up
            cstate.lin_vel_x = min(cstate.lin_vel_x + DV, VMAX)
            print(f"[CTRL] vx={cstate.lin_vel_x:+.2f}")
        elif keycode == 264:  # Down
            cstate.lin_vel_x = max(cstate.lin_vel_x - DV, -VMAX)
            print(f"[CTRL] vx={cstate.lin_vel_x:+.2f}")
        elif keycode == 263:  # Left
            cstate.ang_vel_yaw = min(cstate.ang_vel_yaw + DW, WMAX)
            print(f"[CTRL] yaw={cstate.ang_vel_yaw:+.2f}")
        elif keycode == 262:  # Right
            cstate.ang_vel_yaw = max(cstate.ang_vel_yaw - DW, -WMAX)
            print(f"[CTRL] yaw={cstate.ang_vel_yaw:+.2f}")
        elif keycode == 44:   # ,
            cstate.lin_vel_y = min(cstate.lin_vel_y + DV, VMAX)
            print(f"[CTRL] vy={cstate.lin_vel_y:+.2f}")
        elif keycode == 46:   # .
            cstate.lin_vel_y = max(cstate.lin_vel_y - DV, -VMAX)
            print(f"[CTRL] vy={cstate.lin_vel_y:+.2f}")
        elif keycode == 48:   # 0
            cstate.lin_vel_x = cstate.lin_vel_y = cstate.ang_vel_yaw = 0.0
            print("[CTRL] STOP")
        elif keycode == 55:   # 7
            cstate.gait_name = "bound"
            print("[CTRL] gait=bound")
        elif keycode == 56:   # 8
            cstate.gait_name = "trot"
            print("[CTRL] gait=trot")
        elif keycode == 57:   # 9
            cstate.gait_name = "pace"
            print("[CTRL] gait=pace")
        elif keycode == 80:   # P — screenshot
            cstate.screenshot_requested = True
            print("[CTRL] screenshot queued")
        elif keycode == 82:   # R
            cstate.reset_requested = True
            print("[CTRL] reset")
    return cb


def main():
    print(f"Scene: {SCENE_PATH}")
    print("Loading model and policy...")
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)
    reset_robot(model, data)

    body_net = torch.jit.load(f"{POLICY_DIR}/body_latest.jit")
    body_net.eval()
    adapt_net = torch.jit.load(f"{POLICY_DIR}/adaptation_module_latest.jit")
    adapt_net.eval()

    cstate = ControlState()
    hist = deque([np.zeros(OBS_DIM, dtype=np.float32)
                  for _ in range(HISTORY_LEN)], maxlen=HISTORY_LEN)
    last_a = np.zeros(12)
    prev_a = np.zeros(12)
    targets = DEFAULT_JOINT_POS.copy()
    phase_t = 0.0
    asc = np.full(12, ACTION_SCALE)
    for h in [0, 3, 6, 9]:
        asc[h] *= HIP_SCALE_REDUCTION

    print("\n" + "=" * 58)
    print(f"VIEWER — {scene_name}")
    print("=" * 58)
    print("  Up/Down: fwd/back | Left/Right: yaw | ,/. : strafe")
    print("  0: stop | 8/9/7: trot/pace/bound | R: reset")
    print("  Screenshot: viewer File>Screenshot button (saves PNG to cwd)")
    print("  QUIT: Ctrl+C in this terminal (runs until you do)")
    print("=" * 58 + "\n")

    sim_step = 0
    try:
        with mujoco.viewer.launch_passive(
                model, data, key_callback=make_key_callback(cstate)) as v:
            while v.is_running():   # no time limit — runs until Ctrl+C / close
                if cstate.reset_requested:
                    reset_robot(model, data)
                    hist.clear()
                    for _ in range(HISTORY_LEN):
                        hist.append(np.zeros(OBS_DIM, dtype=np.float32))
                    last_a[:] = 0
                    prev_a[:] = 0
                    targets[:] = DEFAULT_JOINT_POS
                    phase_t = 0.0
                    cstate.reset_requested = False

                if sim_step % DECIMATION == 0:
                    phase_t = (phase_t + STEP_FREQUENCY
                               * model.opt.timestep * DECIMATION) % 1.0
                    obs = build_obs(data, cstate, prev_a, last_a, phase_t)
                    hist.append(obs)
                    ht = torch.tensor(
                        np.concatenate(hist).reshape(1, -1),
                        dtype=torch.float32)
                    with torch.no_grad():
                        lat = adapt_net(ht)
                        act = body_net(
                            torch.cat([ht, lat], dim=1)).numpy().flatten()
                    prev_a = last_a.copy()
                    last_a = act.copy()
                    targets = DEFAULT_JOINT_POS + act * asc

                data.ctrl[:] = (KP * (targets - data.qpos[7:19])
                                - KD * data.qvel[6:18])
                mujoco.mj_step(model, data)
                sim_step += 1
                v.sync()
                if cstate.screenshot_requested:
                    cstate.screenshot_requested = False
                    r = mujoco.Renderer(model, height=1080, width=1920)
                    r.update_scene(data)
                    frame = r.render()
                    r.close()
                    ts = datetime.datetime.now().strftime("%H%M%S")
                    base = scene_name.replace(".xml", "")
                    fn = os.path.join(SHOT_DIR, f"{base}_{ts}.png")
                    imageio.imwrite(fn, frame)
                    print(f"[SHOT] saved -> {fn}")
                # real-time pacing
                time.sleep(max(0.0, model.opt.timestep))
    except KeyboardInterrupt:
        print("\nCtrl+C received — shutting down viewer.")

    print("Done.")


if __name__ == "__main__":
    main()