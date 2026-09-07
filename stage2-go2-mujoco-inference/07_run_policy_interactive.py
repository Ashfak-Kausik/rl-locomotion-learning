"""
Stage 2.6: Interactive policy control with keyboard input.

Same inference pipeline as 06_run_policy.py, but with:
- Real-time keyboard control via MuJoCo viewer key callbacks
- Gait switching (trot / pace / bound)
- Reset functionality
- Cleaner code organization
- Diagnostic overlay (printed every second)

KEY MAPPING (active while viewer window is focused):
    W / Up arrow      Increase forward velocity
    S / Down arrow    Decrease forward velocity
    A / Left arrow    Yaw rate += 0.1 (turn left)
    D / Right arrow   Yaw rate -= 0.1 (turn right)
    Q                 Strafe left
    E                 Strafe right
    Space             Stop all velocity commands
    1                 Trot gait
    2                 Pace gait
    3                 Bound gait
    R                 Reset robot pose
"""

import os
import mujoco
import mujoco.viewer
import numpy as np
import torch
import time
from collections import deque

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = "/home/user/projects/mujoco_menagerie/unitree_go2/scene.xml"
# Repo-relative default; override with GO2_POLICY_DIR to point at a
# different checkpoint directory (e.g. one restored by download_policy.sh).
POLICY_DIR = os.environ.get(
    "GO2_POLICY_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints"),
)

# Joint defaults from walk-these-ways training config (signs verified)
DEFAULT_JOINT_POS = np.array([
     0.1, 0.8, -1.5,   # FL
    -0.1, 0.8, -1.5,   # FR
     0.1, 1.0, -1.5,   # RL
    -0.1, 1.0, -1.5,   # RR
])

OBS_SCALES = {
    "lin_vel": 2.0,            "ang_vel": 0.25,
    "dof_pos": 1.0,            "dof_vel": 0.05,
    "body_height_cmd": 2.0,
    "gait_phase_cmd": 1.0,     "gait_freq_cmd": 1.0,
    "footswing_height_cmd": 0.15,
    "body_pitch_cmd": 0.3,     "body_roll_cmd": 0.3,
    "stance_width_cmd": 1.0,   "stance_length_cmd": 1.0,
    "aux_reward_cmd": 1.0,
}

ACTION_SCALE = 0.25
HIP_SCALE_REDUCTION = 0.5
KP, KD = 25.0, 0.6
DECIMATION = 10
HISTORY_LEN = 30
OBS_DIM = 70

# ============================================================================
# GAIT PRESETS — each is (gait_phase, gait_offset, gait_bound)
# ============================================================================
GAIT_PRESETS = {
    "trot":  (0.5, 0.0, 0.0),  # diagonal pairs (FR+RL together, FL+RR together)
    "pace":  (0.0, 0.5, 0.0),  # lateral pairs (left side together, right side together)
    "bound": (0.0, 0.0, 0.5),  # front pair / rear pair
}
DEFAULT_GAIT = "trot"

# ============================================================================
# CONTROL STATE (mutable, modified by keyboard callbacks)
# ============================================================================
class ControlState:
    """Holds the user-commanded inputs. Updated by keyboard callbacks."""
    def __init__(self):
        self.lin_vel_x = 0.0
        self.lin_vel_y = 0.0
        self.ang_vel_yaw = 0.0
        self.gait_name = DEFAULT_GAIT
        self.reset_requested = False

    @property
    def gait_params(self):
        """Returns (phase, offset, bound) for the current gait."""
        return GAIT_PRESETS[self.gait_name]

    def build_command_vector(self):
        """Assemble the 15-dim command vector for the policy."""
        phase, offset, bound = self.gait_params
        return np.array([
            self.lin_vel_x,     # 0: lin_vel_x
            self.lin_vel_y,     # 1: lin_vel_y
            self.ang_vel_yaw,   # 2: ang_vel_yaw
            0.0,                # 3: body_height
            2.0,                # 4: step_frequency
            phase,              # 5: gait_phase
            offset,             # 6: gait_offset
            bound,              # 7: gait_bound
            0.5,                # 8: gait_duration
            0.06,               # 9: footswing_height
            0.0,                # 10: body_pitch
            0.0,                # 11: body_roll
            0.0,                # 12: stance_width
            0.0,                # 13: stance_length
            0.0,                # 14: aux_reward
        ])


COMMANDS_SCALE = np.array([
    OBS_SCALES["lin_vel"], OBS_SCALES["lin_vel"], OBS_SCALES["ang_vel"],
    OBS_SCALES["body_height_cmd"], OBS_SCALES["gait_freq_cmd"],
    OBS_SCALES["gait_phase_cmd"], OBS_SCALES["gait_phase_cmd"],
    OBS_SCALES["gait_phase_cmd"], OBS_SCALES["gait_phase_cmd"],
    OBS_SCALES["footswing_height_cmd"],
    OBS_SCALES["body_pitch_cmd"], OBS_SCALES["body_roll_cmd"],
    OBS_SCALES["stance_width_cmd"], OBS_SCALES["stance_length_cmd"],
    OBS_SCALES["aux_reward_cmd"],
])

# ============================================================================
# HELPERS
# ============================================================================
def quat_rotate_inverse(q, v):
    q_w = q[0]
    q_vec = q[1:]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (2.0 * q_w)
    c = q_vec * 2.0 * (q_vec @ v)
    return a - b + c


def build_obs(data, ctrl_state, prev_action, last_action, gait_phase_t):
    """Build the 70-dim observation vector."""
    base_quat = data.qpos[3:7]
    gravity_world = np.array([0.0, 0.0, -1.0])
    projected_grav = quat_rotate_inverse(base_quat, gravity_world)

    cmd = ctrl_state.build_command_vector() * COMMANDS_SCALE

    joint_pos = data.qpos[7:19]
    joint_vel = data.qvel[6:18]
    joint_pos_obs = (joint_pos - DEFAULT_JOINT_POS) * OBS_SCALES["dof_pos"]
    joint_vel_obs = joint_vel * OBS_SCALES["dof_vel"]

    # Per-foot phase logic (matches training)
    phase_cmd, offset_cmd, bound_cmd = ctrl_state.gait_params
    foot_phases = np.array([
        gait_phase_t + phase_cmd + offset_cmd + bound_cmd,
        gait_phase_t + offset_cmd,
        gait_phase_t + bound_cmd,
        gait_phase_t + phase_cmd,
    ]) % 1.0
    clock = np.sin(2 * np.pi * foot_phases)

    obs = np.concatenate([
        projected_grav, cmd, joint_pos_obs, joint_vel_obs,
        last_action, prev_action, clock,
    ])
    return obs


def reset_robot(model, data):
    """Reset the robot to the training default pose."""
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    data.qpos[7:19] = DEFAULT_JOINT_POS
    data.qpos[2] = 0.30
    mujoco.mj_forward(model, data)


# ============================================================================
# KEYBOARD CALLBACK
# ============================================================================
def make_key_callback(ctrl_state):
    """
    Key callback bound to the given control state.

    NOTE: Letter keys (W/A/S/D/Q/E/etc.) are reserved by MuJoCo's viewer
    for wireframe, transparency, and other rendering shortcuts. We use
    arrow keys, comma/period, and number keys to avoid conflicts.
    """
    DV = 0.25     # m/s per press — bumped for noticeable response
    DW = 0.25     # rad/s per press
    V_MAX = 1.5
    W_MAX = 1.5

    def key_callback(keycode):
        # Arrow keys
        if keycode == 265:  # UP — forward
            ctrl_state.lin_vel_x = min(ctrl_state.lin_vel_x + DV, V_MAX)
            print(f"[CTRL] lin_vel_x = {ctrl_state.lin_vel_x:+.2f} m/s")
        elif keycode == 264:  # DOWN — backward
            ctrl_state.lin_vel_x = max(ctrl_state.lin_vel_x - DV, -V_MAX)
            print(f"[CTRL] lin_vel_x = {ctrl_state.lin_vel_x:+.2f} m/s")
        elif keycode == 263:  # LEFT — yaw left
            ctrl_state.ang_vel_yaw = min(ctrl_state.ang_vel_yaw + DW, W_MAX)
            print(f"[CTRL] ang_vel_yaw = {ctrl_state.ang_vel_yaw:+.2f} rad/s")
        elif keycode == 262:  # RIGHT — yaw right
            ctrl_state.ang_vel_yaw = max(ctrl_state.ang_vel_yaw - DW, -W_MAX)
            print(f"[CTRL] ang_vel_yaw = {ctrl_state.ang_vel_yaw:+.2f} rad/s")

        # Comma / period for strafe
        elif keycode == 44:  # ,
            ctrl_state.lin_vel_y = min(ctrl_state.lin_vel_y + DV, V_MAX)
            print(f"[CTRL] lin_vel_y = {ctrl_state.lin_vel_y:+.2f} m/s")
        elif keycode == 46:  # .
            ctrl_state.lin_vel_y = max(ctrl_state.lin_vel_y - DV, -W_MAX)
            print(f"[CTRL] lin_vel_y = {ctrl_state.lin_vel_y:+.2f} m/s")

        # Stop — number 0
        elif keycode == 48:  # 0
            ctrl_state.lin_vel_x = 0.0
            ctrl_state.lin_vel_y = 0.0
            ctrl_state.ang_vel_yaw = 0.0
            print("[CTRL] STOP — all velocity commands zeroed")

        # Gait switching — keys 7, 8, 9
        elif keycode == 55:  # 7 — bound (warning)
            ctrl_state.gait_name = "bound"
            print("[CTRL] Gait → bound (WARNING: unstable in MuJoCo)")
        elif keycode == 56:  # 8 — trot (default, most stable)
            ctrl_state.gait_name = "trot"
            print("[CTRL] Gait → trot")
        elif keycode == 57:  # 9 — pace
            ctrl_state.gait_name = "pace"
            print("[CTRL] Gait → pace (may be unstable)")

        # Reset — R is safe (not a MuJoCo viewer shortcut)
        elif keycode == 82:  # R
            ctrl_state.reset_requested = True
            print("[CTRL] Reset requested")

    return key_callback


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("Loading MuJoCo model...")
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    reset_robot(model, data)

    print("Loading policy...")
    body_net = torch.jit.load(f"{POLICY_DIR}/body_latest.jit")
    adapt_net = torch.jit.load(f"{POLICY_DIR}/adaptation_module_latest.jit")
    body_net.eval()
    adapt_net.eval()

    # Initial state
    ctrl_state = ControlState()
    obs_history = deque(
        [np.zeros(OBS_DIM, dtype=np.float32) for _ in range(HISTORY_LEN)],
        maxlen=HISTORY_LEN,
    )
    last_action = np.zeros(12)
    prev_action = np.zeros(12)
    joint_targets = DEFAULT_JOINT_POS.copy()
    gait_phase_t = 0.0
    GAIT_FREQ = 2.0  # step_frequency from command vector
    action_scale_per_joint = np.full(12, ACTION_SCALE)
    for hip_idx in [0, 3, 6, 9]:
        action_scale_per_joint[hip_idx] *= HIP_SCALE_REDUCTION

    print("\n" + "=" * 60)
    print("INTERACTIVE GO2 CONTROL")
    print("=" * 60)
    print("  ↑ / ↓     : forward / backward")
    print("  ← / →     : turn left / right")
    print("  , / .     : strafe left / right")
    print("  0         : stop all commands")
    print("  8         : trot gait (default, stable)")
    print("  9         : pace gait (may be unstable)")
    print("  7         : bound gait (unstable in MuJoCo)")
    print("  R         : reset robot")
    print("  ESC       : quit")
    print()
    print("  NOTE: Letter keys (W/A/S/D etc) are reserved by")
    print("  MuJoCo's viewer for rendering shortcuts (wireframe,")
    print("  transparency, etc). Use arrow keys for movement.")
    print("=" * 60)
    print()

    key_callback = make_key_callback(ctrl_state)

    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback
    ) as viewer:
        sim_step = 0
        start_time = time.time()
        t_last_print = 0.0

        while viewer.is_running():
            step_start = time.time()

            # Handle reset
            if ctrl_state.reset_requested:
                reset_robot(model, data)
                obs_history.clear()
                for _ in range(HISTORY_LEN):
                    obs_history.append(np.zeros(OBS_DIM, dtype=np.float32))
                last_action[:] = 0
                prev_action[:] = 0
                joint_targets[:] = DEFAULT_JOINT_POS
                gait_phase_t = 0.0
                ctrl_state.reset_requested = False

            # Policy step every DECIMATION sim steps
            if sim_step % DECIMATION == 0:
                dt_policy = model.opt.timestep * DECIMATION
                gait_phase_t = (gait_phase_t + GAIT_FREQ * dt_policy) % 1.0

                obs = build_obs(data, ctrl_state, prev_action, last_action, gait_phase_t)
                obs_history.append(obs)

                history_tensor = torch.tensor(
                    np.concatenate(obs_history).reshape(1, -1),
                    dtype=torch.float32,
                )

                with torch.no_grad():
                    env_latent = adapt_net(history_tensor)
                    body_input = torch.cat([history_tensor, env_latent], dim=1)
                    action_delta = body_net(body_input).numpy().flatten()

                prev_action = last_action.copy()
                last_action = action_delta.copy()
                joint_targets = DEFAULT_JOINT_POS + action_delta * action_scale_per_joint

            # Low-level PD control
            q = data.qpos[7:19]
            qd = data.qvel[6:18]
            data.ctrl[:] = KP * (joint_targets - q) - KD * qd

            mujoco.mj_step(model, data)

            # Status print every second
            if data.time - t_last_print >= 1.0:
                base_pos = data.qpos[0:3]
                base_lin_vel = data.qvel[0:3]
                print(
                    f"t={data.time:6.2f}s | "
                    f"gait={ctrl_state.gait_name:5s} | "
                    f"cmd=({ctrl_state.lin_vel_x:+.2f}, {ctrl_state.lin_vel_y:+.2f}, "
                    f"{ctrl_state.ang_vel_yaw:+.2f}) | "
                    f"vel=({base_lin_vel[0]:+.2f}, {base_lin_vel[1]:+.2f}) | "
                    f"h={base_pos[2]:.3f}m"
                )
                t_last_print = data.time

            sim_step += 1
            viewer.sync()

            time_until_next_step = (sim_step * model.opt.timestep) - (time.time() - start_time)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    print("\nDone.")


if __name__ == "__main__":
    main()