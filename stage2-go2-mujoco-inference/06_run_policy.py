"""
Stage 2.5 (continued): Loading the policy and running the sim in MuJoCo.
Goal: Get the Go2 attempting to walk under control of the real policy.

Architecture:
    - Maintain a rolling buffer of last 30 obs (each 70-dim) = 2100 dims total.
    - Each policy step:
        1. Build current 70-dim obs from MuJoCo state (correct layout).
        2. Update the rolling history.
        3. Pass history through adapatation module -> 2-dim env latent.
        4. Concatenate history + env latent -> 2102-dim input to body.
        5. Body outputs 12-dim action (joint position deltas).
        6. Action -> joint targets -> PD controller -> motor torques. 

Policy runs at 50 Hz (every 10th physics step at 0.002s timestep).        
"""

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import MODEL_PATH, POLICY_DIR  # env-overridable, see paths.py

import mujoco 
import mujoco.viewer
import numpy as np
import torch
import time
from collections import deque

# PATHS and CONSTANTS

# Training using these exact values - must match for sim-to-sim transfer. 
DEFAULT_JOINT_POS = np.array([
     0.1, 0.8, -1.5,   # FL
    -0.1, 0.8, -1.5,   # FR
     0.1, 1.0, -1.5,   # RL
    -0.1, 1.0, -1.5,   # RR
])

# From our training config dump:
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

# Action scale; policy outputs deltas, scaled before adding to default pose.
# This is the value 'action_scale' from our training config: 0.25
ACTION_SCALE = 0.25
HIP_SCALE_REDUCTION = 0.5  # hips get smaller deltas (more conservative).

# PD gains for low-level torque control (from training config) 
KP = 25.0
KD = 0.6

# Control: policy at 50 Hz, sim at 500 Hz -> decimation factor of 10.
# sim runs at 500 Hz (0.002s timestep)
DECIMATION = 10

# History length for adaptation module:
HISTORY_LEN = 30  # timesteps
OBS_DIM = 70     # dims per timestep

# COMMANDS - what we'd "ask" the robot to do.

# Explicit trot command:
COMMANDS = np.array([
    0.5,   # 0: lin_vel_x (m/s)
    0.0,   # 1: lin_vel_y (m/s)
    0.0,   # 2: ang_vel_yaw (rad/s)
    0.0,   # 3: body_height (m)
    2.0,   # 4: step_frequency (Hz): lower to 2 Hz for cleaner trot
    0.5,   # 5: gait_phase (Hz): 0.5 = trot (diagonal pairs)
    0.0,   # 6: gait_offset
    0.0,   # 7: gait_bound: keep at 0 (non-zero triggers bound)
    0.5,   # 8: gait_duration (s)
    0.06,  # 9: footswing_height (m): lower slightly
    0.0,   # 10: body_pitch (rad)
    0.0,   # 11: body_roll (rad)
    0.0,   # 12: stance_width (m)
    0.0,   # 13: stance_length (m)
    0.0,   # 14: aux_reward
])

# Scale each command by its corresponding obs_scale
COMMANDS_SCALE = np.array([
    OBS_SCALES["lin_vel"],
    OBS_SCALES["lin_vel"],
    OBS_SCALES["ang_vel"],
    OBS_SCALES["body_height_cmd"],
    OBS_SCALES["gait_freq_cmd"],
    OBS_SCALES["gait_phase_cmd"],
    OBS_SCALES["gait_phase_cmd"],
    OBS_SCALES["gait_phase_cmd"],
    OBS_SCALES["gait_phase_cmd"],
    OBS_SCALES["footswing_height_cmd"],
    OBS_SCALES["body_pitch_cmd"],
    OBS_SCALES["body_roll_cmd"],
    OBS_SCALES["stance_width_cmd"],
    OBS_SCALES["stance_length_cmd"],
    OBS_SCALES["aux_reward_cmd"],
])

# HELPERS: for building the obs vector and computing actions from policy output.
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

def build_obs(data, prev_action, last_action, gait_phase_t):
    """
    Build the 70-dim observation in the EXACT order the policy was trained with:
       [0:3]   projected_gravity
       [3:18]  commands * commands_scale
       [18:30] (joint_pos - default) * scale
       [30:42] joint_vel * scale
       [42:54] current action (last action sent to robot)
       [54:66] previous action (the one before that)
       [66:70] clock signals
    """

    # Projected gravity in body frame:
    base_quat = data.qpos[3:7]                        # [w, x, y, z]
    gravity_world = np.array([0.0, 0.0, -1.0])        # gravity vector in world frame
    projected_grav = quat_rotate_inverse(base_quat, gravity_world)

    # Scaled commands 
    cmd = COMMANDS * COMMANDS_SCALE

    # Joint state
    joint_pos = data.qpos[7:19]
    joint_vel = data.qvel[6:18]
    joint_pos_obs = (joint_pos - DEFAULT_JOINT_POS) * OBS_SCALES["dof_pos"]
    joint_vel_obs = joint_vel * OBS_SCALES["dof_vel"]

    # Per-foot phases, derived as in training (see legged_robot.py line ~720)
    # Foot 0: gait_phase_t + phases + offsets + bounds
    # foot 1: gait_phase_t + offsets
    # foot 2: gait_phase_t + bounds
    # foot 3: gait_phase_t + phases
    phases = COMMANDS[5]               # gait_phase_cmd
    offsets = COMMANDS[6]              # gait_offset
    bounds = COMMANDS[7]               # gait_bound
    foot_phases = np.array([
        (gait_phase_t + phases + offsets + bounds),
        gait_phase_t + offsets,
        gait_phase_t + bounds,
        gait_phase_t + phases,
    ]) % 1.0                           # wrap to [0, 1]

    # Note: With gate_duration = 0.5 (default), the stance/swing remapping is the identity, so we skip it. If duration changes, must add remapping.

    clock = np.sin(2 * np.pi * foot_phases)  # convert to clock signals for each foot

    obs = np.concatenate([
        projected_grav,    # 3
        cmd,               # 15
        joint_pos_obs,     # 12
        joint_vel_obs,     # 12
        last_action,       # 12  (most recent action sent to robot)
        prev_action,       # 12  (previous action sent to robot)
        clock,             # 4   (clock signals)
    ])
    assert obs.shape == (70,), f"obs shape is {obs.shape}, expected (70,)"
    return obs

# SETUP MUJOCO
print("Loading MuJoCo model...")
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(model, data, key_id)

# Override the keyframe joints with the training default pose
# (the menagerie keyframe uses different angles than walk-these-ways training)
data.qpos[7:19] = DEFAULT_JOINT_POS
data.qpos[2] = 0.30  # also set body height
mujoco.mj_forward(model, data)

print("\n=== Actuator joint order ===")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"  ctrl[{i}] -> {name}")
print()

print(f"Sim timestep: {model.opt.timestep}s, control frequency: {1/(model.opt.timestep*DECIMATION)} Hz")
print(f"Decimation: {DECIMATION} -> policy at {1/(model.opt.timestep*DECIMATION)} Hz")

print("Loading policy...")
body_net = torch.jit.load(f"{POLICY_DIR}/body_latest.jit")
adapt_net = torch.jit.load(f"{POLICY_DIR}/adaptation_module_latest.jit")
body_net.eval()
adapt_net.eval()

# History buffer - 30 frames of 70-dim obs
obs_history = deque([np.zeros(OBS_DIM, dtype=np.float32) for _ in range(HISTORY_LEN)], 
                    maxlen=HISTORY_LEN)

# Action state 
last_action = np.zeros(12)                       # the most recent action sent to the robot (for obs)
prev_action = np.zeros(12)                       # the action before that (for obs)
current_joint_pos = DEFAULT_JOINT_POS.copy()     # track current joint positions for PD control

# Gait clock
gait_phase_t = 0.0                               # tracks the current phase of the gait cycle, from 0 to 1.
GAIT_FREQ = COMMANDS[4]                          # step frequency in Hz (from commands) 

# Hip-scale-reduction mask (FR_hip, FL_hip, RR_hip, RL_hip are indices 0, 3, 6, 9 in the action vector)
action_scale_per_joint = np.full(12, ACTION_SCALE)
for hip_idx in [0, 3, 6, 9]:
    action_scale_per_joint[hip_idx] *= HIP_SCALE_REDUCTION


# RUN (THE MAIN SIMULATION)
print("\nLaunching MuJoCo viewer. Robot should attempt to walk forward at ~0.5 m/s under control of the loaded policy.\n")
print("Press Esc or close the window to quit.")

with mujoco.viewer.launch_passive(model, data) as viewer:
    sim_step = 0
    start_time = time.time()
    t_last_print = 0.0 

    while viewer.is_running() and time.time() - start_time < 60:      # run for 30 seconds or until window is closed.
        step_start = time.time()

        # Policy step every DECIMATION sim steps
        if sim_step % DECIMATION == 0:
            # Update gait clock
            dt_policy = model.opt.timestep * DECIMATION
            gait_phase_t = (gait_phase_t + GAIT_FREQ * dt_policy) % 1.0

            # Build current obs and update history
            obs = build_obs(data, prev_action, last_action, gait_phase_t)
            obs_history.append(obs)

            # Flatten history into (1, 2100) tensor
            history_tensor = torch.tensor(
                np.concatenate(obs_history).reshape(1, -1), 
                dtype=torch.float32,
            )

            # Run adaptation module to get env latent, then body net to get action
            with torch.no_grad():
                env_latent = adapt_net(history_tensor)                       # (1, 2)
                body_input = torch.cat([history_tensor, env_latent], dim=1)  # (1, 2102)
                action_delta = body_net(body_input).numpy().flatten()        # (12,)

            # Shift action history
            prev_action = last_action.copy()
            last_action = action_delta.copy()

            # Convert action to joint targets
            joint_targets = DEFAULT_JOINT_POS + action_delta * action_scale_per_joint

        
        # --- Low-level PD control every sim step ---
        q = data.qpos[7:19]   # current joint positions
        qd = data.qvel[6:18]  # current joint velocities
        data.ctrl[:] = KP * (joint_targets - q) - KD * qd

        mujoco.mj_step(model, data)

        # --- Print state once per second ---
        if data.time - t_last_print >= 1.0:
            base_pos = data.qpos[0:3]
            base_lin_vel = data.qvel[0:3]
            print(f"Time: {data.time:.2f}s | Base pos: {base_pos} | Base lin vel: {base_lin_vel}")
            t_last_print = data.time

        sim_step += 1
        viewer.sync()  # sync to real time (will slow down if sim is running too fast)

        # Real-time pacing
        time_until_next_step = (sim_step * model.opt.timestep) - (time.time() - start_time)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)    


print("Done! Simulation ended.")