"""
Stage 2.1: Loading the Go2 model in MuJoco and opening a viewer window
Goal: Let's see the Go2 robot in the MuJoCo sim environment.
"""

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import MODEL_PATH  # env-overridable, see paths.py

import mujoco
import mujoco.viewer
import time

# Path to the Go2 scene XML file from the MuJoCo Menagerie.

# Load the model (This parses the XML file and builds the physics model in memory).
model = mujoco.MjModel.from_xml_path(MODEL_PATH)

# Create a "data" object (this holds the current simulation state).
data = mujoco.MjData(model)                          # Joint positions, velocities, forces, etc. are stored here (Model is static but data is dynamic).

# Launch the viewer to visualize the simulation (a window that renders the current state of the simulation).
# "passive" means We control the simulation step manually (the viewer just displays the current state).
with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()                              # Run for 30 seconds of real time.
    while viewer.is_running() and time.time() - start < 30:
        step_start = time.time()                     # Record the start time of this simulation step.   
        mujoco.mj_step(model, data)           
        viewer.sync()                               # Update the viewer with the latest simulation state.

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)          # Sleep to maintain real-time pacing (simulate at real-time speed).

print("Done and Simulation ended.")