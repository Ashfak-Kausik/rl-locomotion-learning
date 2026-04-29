"""
Stage 2.4: Building the policy's observation vector by hand. 
Goal: Let's understand what the policy actually "sees" - extract each piece from MuJoCo state, scale it, and assemble a single 70-dim vector. 

We won't run a policy yet. Just constructing one and inspecting the obs.
"""

import mujoco 
import mujoco.viewer
import numpy as np
import time

MODEL_PATH = "/home/user/projects/mujoco_menagerie/unitree_go2/scene.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Reset to the standing keyframe pose (same as before) so we start from a known pose.
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(model, data, key_id)

# Default joint pose - joint angles are observed RELATIVE to this pose.
# This must match the default_joint_angles used during training. 
DEFAULT_JOINT_POS = np.array([
    -0.1, 0.8, -1.5,  # Front Left leg (hip, thigh, calf)
     0.1, 0.8, -1.5,  # Front Right leg (hip, thigh, calf)
    -0.1, 1.0, -1.5,  # Rear Left leg (hip, thigh, calf)
     0.1, 1.0, -1.5   # Rear Right leg (hip, thigh, calf)
])

# Observation scales - pulled from our training config dump.
# These exist because neural networks train better when inputs are roughly in the same range (e.g. -1 to 1).
# Joint velocities can be +/- 20 rad/s, which is much bigger than joint positions in radians, so we scale them down.
OBS_SCALES = {
    "lin_vel": 2.0,
    "ang_vel": 0.25,
    "dof_pos": 1.0,
    "dof_vel": 0.05,
    "body_height_cmd": 2.0,
    "gait_phase_cmd": 1.0,
    "gait_freq_cmd": 1.0,
    "footswing_height_cmd": 0.15,
    "body_pitch_cmd": 0.3,
    "body_roll_cmd": 0.3,
    "stance_width_cmd": 1.0,
    "stance_length_cmd": 1.0,
    "aux_reward_cmd": 1.0,
}

# Commands the user/joystick would send.
# These are what we'd "ask" the robot to do. For now: stand still.
COMMANDS = {
    "lin_vel_x": 0.0,   # forward/backward speed (m/s)
    "lin_vel_y": 0.0,   # left/right speed (m/s)
    "ang_vel_yaw": 0.0, # turning speed (rad/s)
    "body_height": 0.0, # height adjustment from default (m)
    "step_frequency": 3.0, # how fast to step (Hz)
    "gait_phase": 0.5, # where we are in the gait cycle (0 to 1)
    "gait_offset": 0.0,  # phase offset for leg timing (rad)
    "gait_bound": 0.0,   # how much to lift feet during swing (m)
    "gait_duration": 0.5, # how long each step should last (s)
    "footswing_height": 0.08, # how high to lift feet during swing (m)
    "body_pitch": 0.0,  # desired body pitch angle (rad)
    "body_roll": 0.0,   # desired body roll angle (rad)
    "stance_width": 0.0, # desired change in stance width (m)
    "stance_length": 0.0, # desired change in stance length (m) 
    "aux_reward": 0.0,   # extra reward signal (for training, not used in control)
}

def quat_rotate_inverse(q, v):
    """
    Rotate a vector v by the inverse of a quaternion q.
    This is used to express world-frame quantities in body frame.
    q = [w, x, y, z], v = [x, y, z]
    """

    q_w = q[0]
    q_vec = q[1:]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (2.0 * q_w)
    c = q_vec * 2.0 * (q_vec @ v)
    return a - b + c

# Start with zeros for "previous action" - the first frame has no prior action.
prev_action = np.zeros(12)
prev_prev_action = np.zeros(12)

# Gait clock - accumulates over time
gait_phase_t = 0.0

def build_obs(model, data, commands, prev_action, prev_prev_action, gait_phase_t):
    """Construct the 70 dim observation vector that the policy will see at each timestep."""

    # ===[0:3] Projected gravity in body frame (3 values)===
    # In the world frame, gravity always points down: [0, 0, -1] (after normalization).
    # But the policy needs to know which way is down RALTIVE to the ROBOT's body.
    # So we rotate the world-frame gravity vector into the body's frame using the inverse of the body's orientation quaternion.
    base_quat = data.qpos[3:7]                        # [w, x, y, z]
    gravity_world = np.array([0.0, 0.0, -1.0])        # gravity vector in world frame
    projected_gravity = quat_rotate_inverse(base_quat, gravity_world)

    # == [3:6] Velocity commands (scaled) ==
    cmd_vel = np.array([
        COMMANDS["lin_vel_x"] * OBS_SCALES["lin_vel"],
        COMMANDS["lin_vel_y"] * OBS_SCALES["lin_vel"],
        COMMANDS["ang_vel_yaw"] * OBS_SCALES["ang_vel"],
    ])

    # ===[6:18] Joint positions (relative to default pose, scaled)===
    joint_pos = data.qpos[7:19]
    joint_pos_obs = (joint_pos - DEFAULT_JOINT_POS) * OBS_SCALES["dof_pos"]

    # == [18:30] Joint velocities (scaled) ==
    joint_vel = data.qvel[6:18]
    joint_vel_obs = joint_vel * OBS_SCALES["dof_vel"]

    # ===[30:42] Previous action (scaled)===
    # The action the policy output last step. Helps the policy be temporally coherent - it can "remember" what it just did.
    action_obs = prev_action.copy()

    # ===[42:54] clock signals ===
    # Two phase signals split into front/reaer or diagonal pairs.
    # For trot; feet move in diagonal pairs. Phase shifted by 0.5 between pairs. 
    # Encoded as sin/cos so the network sees a continuous presentation of the gait cycle.
    phase = gait_phase_t
    phases = np.array([phase, phase + 0.5, phase + 0.5, phase + 0.5])  # FR, FL, RR, RL
    phases = phases % 1.0  # wrap around at 1.0
    clock_signals = np.array([
    np.sin(2 * np.pi * phases[0]),
    np.sin(2 * np.pi * phases[1]),
    np.sin(2 * np.pi * phases[2]),
    np.sin(2 * np.pi * phases[3]),
    ])

    # === [46:54] Gait commands (8 dims) ===
    # These tell the policy WHAT GAIT to use. walk-these-ways is gait-conditioned. 
    gait_cmd = np.array([
        COMMANDS["step_frequency"] * OBS_SCALES["gait_freq_cmd"],
        COMMANDS["gait_phase"] * OBS_SCALES["gait_phase_cmd"],
        COMMANDS["gait_offset"] * OBS_SCALES["gait_phase_cmd"],
        COMMANDS["gait_bound"] * OBS_SCALES["gait_phase_cmd"],
        COMMANDS["gait_duration"] * OBS_SCALES["gait_phase_cmd"],
        COMMANDS["footswing_height"] * OBS_SCALES["footswing_height_cmd"],
        COMMANDS["body_pitch"] * OBS_SCALES["body_pitch_cmd"],
        COMMANDS["body_roll"] * OBS_SCALES["body_roll_cmd"],
    ])

    # === [54:57] Body shape commands (3 dims) ===
    body_cmd = np.array([
        COMMANDS["body_height"] * OBS_SCALES["body_height_cmd"],
        COMMANDS["stance_width"] * OBS_SCALES["stance_width_cmd"],
        COMMANDS["stance_length"] * OBS_SCALES["stance_length_cmd"],
    ])

    # === [57:70] Last 13 dims: aux + previous-previous action (scaled) ===
    # 1 dim for aux reward command, 12 dim for prev_prev action.
    aux = np.array([COMMANDS["aux_reward"] * OBS_SCALES["aux_reward_cmd"]])
    tail = np.concatenate([aux, prev_prev_action])

    # Concatenate all pieces together into one big observation vector.
    obs = np.concatenate([
        projected_gravity,  # 3
        cmd_vel,            # 3
        joint_pos_obs,      # 12
        joint_vel_obs,      # 12
        action_obs,         # 12
        clock_signals,      # 4
        gait_cmd,           # 8
        body_cmd,           # 3
        tail                # 13
    ])

    return obs

# Run the simulation, hold the standing pose, and print the obs vector once
mujoco.mj_forward(model, data)

standing_target = data.qpos[7:19].copy()
kp, kd = 100.0, 2.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    printed = False
    while viewer.is_running() and time.time() - start < 10:
        step_start = time.time()

        # Hold the standing pose with a PD controller (same as before)
        q = data.qpos[7:19]
        dq = data.qvel[6:18]
        data.ctrl[:] = kp * (standing_target - q) - kd * dq

        mujoco.mj_step(model, data)

        # Build the obs vector and print it once after the robot stabilizes in the standing pose.
        gait_phase_t = (gait_phase_t + COMMANDS["step_frequency"] * model.opt.timestep) % 1.0
        obs = build_obs(model, data, COMMANDS, prev_action, prev_prev_action, gait_phase_t) 

        if data.time > 2.0 and not printed:
            print(f"\n=== Observation vector at time {data.time:.2f}s ===")
            print(f"Total dims: {len(obs)}")
            print(f"\n[0:3] Projected gravity in body frame: {obs[0:3]}")
            print(f"\n[3:6] Velocity commands (cmd_vel): {obs[3:6]}")
            print(f"\n[6:18] Joint positions (joint_pos_rel): {obs[6:18]}")
            print(f"\n[18:30] Joint velocities (joint_vel): {obs[18:30]}")
            print(f"\n[30:42] Previous action (prev_action): {obs[30:42]}")
            print(f"\n[42:46] Clock signals (clock_signals): {obs[42:46]}")
            print(f"\n[46:54] Gait commands (gait_cmd): {obs[46:54]}")
            print(f"\n[54:57] Body shape commands (body_cmd): {obs[54:57]}")
            print(f"\n[57:70] Tail (aux + prev_prev_action): {obs[57:70]}")
            printed = True

        viewer.sync()
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

print("Viewer closed, exiting.")
