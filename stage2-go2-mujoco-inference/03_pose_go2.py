"""
Stage 2.3: Set the Go2 robot to a specific pose (standing pose and hold it there).
Goal: Manually control joints. Understand the difference between qpos (state) and ctrl (commands).

Key insight: The Go2 XML uses raw torque motors, NOT position controllers.
So ctrl = torque (Nm), not target angle. To hold a pose we must implement
a PD controller manually — compute torque based on joint angle error every step.
"""

import mujoco
import mujoco.viewer
import time

MODEL_PATH = "/home/user/projects/mujoco_menagerie/unitree_go2/scene.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Load the officially tuned standing pose from the keyframe defined in go2.xml
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(model, data, key_id)

# Save the target joint angles ONCE before the loop (frozen reference)
# .copy() is critical — without it, this would update as qpos changes
standing_target = data.qpos[7:19].copy()

# PD controller gains
# kp = position gain — how strongly to push joints toward the target angle
# kd = velocity gain — damping to prevent oscillation/wobbling
kp = 100.0
kd = 2.0

# Recompute derived quantities after loading the keyframe
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    while viewer.is_running() and time.time() - start < 30:
        step_start = time.time()

        # Current joint state
        q = data.qpos[7:19]   # current joint angles (12 values)
        dq = data.qvel[6:18]  # current joint velocities (12 values)

        # PD control law: torque = kp * (target - current_angle) - kd * current_velocity
        # This computes how much torque to apply to each joint to drive it toward the target
        data.ctrl[:] = kp * (standing_target - q) - kd * dq

        # Advance physics by one timestep (0.002s = 500Hz)
        mujoco.mj_step(model, data)

        # Print sensor data once per second
        if abs(data.time % 1.0) < model.opt.timestep:
            base_height = data.qpos[2]
            joint_angles = data.qpos[7:19]
            print(f"Time: {data.time:.2f}s | height={base_height:.3f}m | "
                  f"FL_hip={joint_angles[0]:.2f} rad | "
                  f"FL_thigh={joint_angles[1]:.3f} rad")

        # Push the new state to the viewer window
        viewer.sync()

        # Sleep to maintain real-time playback speed
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

print("Viewer closed, exiting.")