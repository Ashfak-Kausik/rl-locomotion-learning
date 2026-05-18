"""Quick visual sanity check of a generated terrain scene (no policy)."""
import sys
import os
import time
import numpy as np
import mujoco
import mujoco.viewer

scene_name = sys.argv[1] if len(sys.argv) > 1 else "go2_slope_05.xml"
scene_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "scenes", scene_name)
)

print(f"Loading: {scene_path}")
m = mujoco.MjModel.from_xml_path(scene_path)
d = mujoco.MjData(m)

key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
if key >= 0:
    mujoco.mj_resetDataKeyframe(m, d, key)

d.qpos[7:19] = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5,
                          0.1, 1.0, -1.5, -0.1, 1.0, -1.5])
d.qpos[2] = 0.30
mujoco.mj_forward(m, d)

print(f"Loaded OK. Robot start z = {d.qpos[2]:.3f} | total geoms = {m.ngeom}")
print("Opening viewer for 10 seconds (no controller — robot will sag).")
print("Look for: flat start area, ramp/steps ahead toward +X, robot resting on ground.")

with mujoco.viewer.launch_passive(m, d) as v:
    t0 = time.time()
    while v.is_running() and time.time() - t0 < 10:
        mujoco.mj_step(m, d)
        v.sync()
        time.sleep(0.002)

print("Visual check done.")
