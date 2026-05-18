"""
Experiment harness for sim-to-sim transfer study.

Core function: run_trial(...) executes one headless trial and returns a
dict of metrics. No viewer — runs as fast as CPU allows.

This is the shared backbone for Experiments 1, 2, and 3.
"""

import os
import numpy as np
import torch
import mujoco
from collections import deque

# ============================================================================
# CONSTANTS (must match the inference scripts exactly)
# ============================================================================
POLICY_DIR = (
    "/home/user/projects/robot-dog-sim/walk-these-ways-go2/runs/"
    "gait-conditioned-agility/pretrain-go2/train/142238.667503/checkpoints"
)

DEFAULT_JOINT_POS = np.array([
     0.1, 0.8, -1.5,   # FL
    -0.1, 0.8, -1.5,   # FR
     0.1, 1.0, -1.5,   # RL
    -0.1, 1.0, -1.5,   # RR
])

OBS_SCALES = {
    "lin_vel": 2.0, "ang_vel": 0.25,
    "dof_pos": 1.0, "dof_vel": 0.05,
    "body_height_cmd": 2.0,
    "gait_phase_cmd": 1.0, "gait_freq_cmd": 1.0,
    "footswing_height_cmd": 0.15,
    "body_pitch_cmd": 0.3, "body_roll_cmd": 0.3,
    "stance_width_cmd": 1.0, "stance_length_cmd": 1.0,
    "aux_reward_cmd": 1.0,
}

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

GAIT_PRESETS = {
    "trot":  (0.5, 0.0, 0.0),
    "pace":  (0.0, 0.5, 0.0),
    "bound": (0.0, 0.0, 0.5),
}

ACTION_SCALE = 0.25
HIP_SCALE_REDUCTION = 0.5
KP, KD = 25.0, 0.6
DECIMATION = 10
HISTORY_LEN = 30
OBS_DIM = 70
STEP_FREQUENCY = 2.0

# Fall thresholds (justified in paper: 0.15m ≈ half nominal standing height;
# 60° ≈ point past which quadruped self-recovery is infeasible)
FALL_HEIGHT_THRESHOLD = 0.15      # meters
FALL_TILT_THRESHOLD_DEG = 60.0    # degrees from upright

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


def tilt_angle_deg(quat):
    """Angle (degrees) between body's up-axis and world up-axis."""
    # Body up vector in world frame: rotate [0,0,1] by quat
    # Using projected gravity is simpler: if upright, proj_grav ≈ [0,0,-1]
    grav_body = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
    # Angle between grav_body and ideal [0,0,-1]
    cos_a = np.clip(-grav_body[2], -1.0, 1.0)  # -z component
    return np.degrees(np.arccos(cos_a))


def build_obs(data, commands_vec, gait_params, prev_action, last_action, gait_phase_t):
    base_quat = data.qpos[3:7]
    proj_grav = quat_rotate_inverse(base_quat, np.array([0.0, 0.0, -1.0]))

    cmd = commands_vec * COMMANDS_SCALE

    joint_pos = data.qpos[7:19]
    joint_vel = data.qvel[6:18]
    joint_pos_obs = (joint_pos - DEFAULT_JOINT_POS) * OBS_SCALES["dof_pos"]
    joint_vel_obs = joint_vel * OBS_SCALES["dof_vel"]

    phase_cmd, offset_cmd, bound_cmd = gait_params
    foot_phases = np.array([
        gait_phase_t + phase_cmd + offset_cmd + bound_cmd,
        gait_phase_t + offset_cmd,
        gait_phase_t + bound_cmd,
        gait_phase_t + phase_cmd,
    ]) % 1.0
    clock = np.sin(2 * np.pi * foot_phases)

    return np.concatenate([
        proj_grav, cmd, joint_pos_obs, joint_vel_obs,
        last_action, prev_action, clock,
    ])


# ============================================================================
# CORE: run one trial
# ============================================================================
def run_trial(scene_path, lin_vel_x=0.5, lin_vel_y=0.0, ang_vel_yaw=0.0,
              gait="trot", settle_s=3.0, measure_s=30.0,
              body_net=None, adapt_net=None, seed=0):
    """
    Run one headless trial. Returns a metrics dict.

    The policy nets can be passed in (to avoid reloading every trial).
    If None, they are loaded here.
    """
    if body_net is None:
        body_net = torch.jit.load(f"{POLICY_DIR}/body_latest.jit")
        body_net.eval()
    if adapt_net is None:
        adapt_net = torch.jit.load(f"{POLICY_DIR}/adaptation_module_latest.jit")
        adapt_net.eval()

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    # Reset to keyframe, override with training default pose
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    data.qpos[7:19] = DEFAULT_JOINT_POS
    data.qpos[2] = 0.30

    # Controlled initial-condition randomization (seed-dependent).
    # Represents real deployment variability: no two robot starts are
    # identical. Makes trials genuinely independent for mean/std stats.
    rng = np.random.default_rng(seed)
    # Small joint angle perturbation (±0.02 rad ≈ ±1.1 deg)
    data.qpos[7:19] += rng.uniform(-0.02, 0.02, size=12)
    # Small base height perturbation (±1 cm)
    data.qpos[2] += rng.uniform(-0.01, 0.01)
    # Small base yaw perturbation (±0.05 rad ≈ ±2.9 deg)
    yaw = rng.uniform(-0.05, 0.05)
    # Apply yaw to the base quaternion (qpos[3:7] = [w,x,y,z])
    half = yaw / 2.0
    data.qpos[3] = np.cos(half)   # w
    data.qpos[4] = 0.0            # x
    data.qpos[5] = 0.0            # y
    data.qpos[6] = np.sin(half)   # z (yaw rotation about vertical)
    # Small initial joint velocity noise
    data.qvel[6:18] += rng.uniform(-0.05, 0.05, size=12)

    mujoco.mj_forward(model, data)

    commands_vec = np.array([
        lin_vel_x, lin_vel_y, ang_vel_yaw, 0.0, STEP_FREQUENCY,
        *GAIT_PRESETS[gait], 0.5, 0.06, 0.0, 0.0, 0.0, 0.0, 0.0,
    ])
    gait_params = GAIT_PRESETS[gait]

    obs_history = deque(
        [np.zeros(OBS_DIM, dtype=np.float32) for _ in range(HISTORY_LEN)],
        maxlen=HISTORY_LEN,
    )
    last_action = np.zeros(12)
    prev_action = np.zeros(12)
    joint_targets = DEFAULT_JOINT_POS.copy()
    gait_phase_t = 0.0

    action_scale_per_joint = np.full(12, ACTION_SCALE)
    for hip_idx in [0, 3, 6, 9]:
        action_scale_per_joint[hip_idx] *= HIP_SCALE_REDUCTION

    dt = model.opt.timestep
    total_s = settle_s + measure_s
    n_steps = int(total_s / dt)
    measure_start_step = int(settle_s / dt)

    # Logging buffers (only during measurement window)
    vx_log, vy_log, h_log = [], [], []
    fell = False
    fall_time = None

    start_pos = None

    for step in range(n_steps):
        if step % DECIMATION == 0:
            gait_phase_t = (gait_phase_t + STEP_FREQUENCY * dt * DECIMATION) % 1.0
            obs = build_obs(data, commands_vec, gait_params,
                            prev_action, last_action, gait_phase_t)
            obs_history.append(obs)
            hist = torch.tensor(
                np.concatenate(obs_history).reshape(1, -1), dtype=torch.float32
            )
            with torch.no_grad():
                latent = adapt_net(hist)
                action = body_net(torch.cat([hist, latent], dim=1)).numpy().flatten()
            prev_action = last_action.copy()
            last_action = action.copy()
            joint_targets = DEFAULT_JOINT_POS + action * action_scale_per_joint

        q = data.qpos[7:19]
        qd = data.qvel[6:18]
        data.ctrl[:] = KP * (joint_targets - q) - KD * qd
        mujoco.mj_step(model, data)

        # Fall check (every step, throughout)
        height = data.qpos[2]
        tilt = tilt_angle_deg(data.qpos[3:7])
        if height < FALL_HEIGHT_THRESHOLD or tilt > FALL_TILT_THRESHOLD_DEG:
            fell = True
            fall_time = step * dt
            break

        # Logging (only in measurement window)
        if step == measure_start_step:
            start_pos = data.qpos[0:3].copy()
        if step >= measure_start_step:
            vx_log.append(data.qvel[0])
            vy_log.append(data.qvel[1])
            h_log.append(data.qpos[2])

    # Compute metrics
    if fell:
        end_pos = data.qpos[0:3].copy()
        return {
            "scene": os.path.basename(scene_path),
            "cmd_vx": lin_vel_x, "cmd_vy": lin_vel_y, "cmd_yaw": ang_vel_yaw,
            "gait": gait, "seed": seed,
            "fell": True, "fall_time_s": round(fall_time, 2),
            "mean_vx": None, "mean_vy": None,
            "vel_track_err": None,
            "lateral_drift": None,
            "height_mean": None, "height_std": None,
            "distance_traveled": None,
        }

    vx_arr = np.array(vx_log)
    vy_arr = np.array(vy_log)
    h_arr = np.array(h_log)
    end_pos = data.qpos[0:3].copy()
    dist = np.linalg.norm(end_pos[:2] - start_pos[:2]) if start_pos is not None else None

    return {
        "scene": os.path.basename(scene_path),
        "cmd_vx": lin_vel_x, "cmd_vy": lin_vel_y, "cmd_yaw": ang_vel_yaw,
        "gait": gait, "seed": seed,
        "fell": False, "fall_time_s": None,
        "mean_vx": round(float(vx_arr.mean()), 4),
        "mean_vy": round(float(vy_arr.mean()), 4),
        "vel_track_err": round(float(abs(lin_vel_x - vx_arr.mean())), 4),
        "lateral_drift": round(float(np.abs(vy_arr).mean()), 4),
        "height_mean": round(float(h_arr.mean()), 4),
        "height_std": round(float(h_arr.std()), 4),
        "distance_traveled": round(float(dist), 4) if dist is not None else None,
    }


if __name__ == "__main__":
    # Smoke test: one trial on flat ground, print the metrics dict
    import json
    scene = os.path.join(os.path.dirname(__file__), "..", "scenes", "go2_flat.xml")
    scene = os.path.abspath(scene)
    print(f"Smoke test on: {scene}")
    result = run_trial(scene, lin_vel_x=0.5, gait="trot",
                       settle_s=3.0, measure_s=10.0, seed=0)
    print(json.dumps(result, indent=2))