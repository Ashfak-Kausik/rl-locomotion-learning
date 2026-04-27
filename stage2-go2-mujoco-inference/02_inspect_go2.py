"""
Stage 2.2: Inspect the Go2 model structure.
Goal: Understand what bodies, joints, actuators, and sensors exist in the Go2 model and how they are organized.
"""

import mujoco

MODEL_PATH = "/home/user/projects/mujoco_menagerie/unitree_go2/scene.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print("=" * 60)                          # This just a sperator, means "print a line of 60 equal signs" to visually separate sections of output.
print(f"Model has {model.nq} generalized positions (qpos)")
print(f"Model has {model.nv} generalized velocities (qvel)")
print(f"Model has {model.nu} actuators (ctrl inputs)")
print (f"Model has {model.nbody} bodies")
print(f"Model has {model.njnt} joints")
print(f"Model has {model.nsensor} sensors")
print("=" * 60) 

print("\n-- Bodies --")
for i in range (model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    print(f"[{i}] {name}")

print("\n-- Joints --")
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    jtype = ["FREE", "BALL", "SLIDE", "HINGE"][model.jnt_type[i]]
    print(f"[{i}] {name} (type: {jtype})")

print("\n-- Actuators --")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    ctrl_range = model.actuator_ctrlrange[i]
    print(f"[{i}] {name} (range: [{ctrl_range[0]:.2f}, {ctrl_range[1]:.2f}])")

print("\n-- Sensors --")
for i in range(model.nsensor):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
    dim = model.sensor_dim[i]
    print(f"[{i}] {name} (dim: {dim})")