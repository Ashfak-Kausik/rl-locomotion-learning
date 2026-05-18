"""
Generate parametric terrain scenes for Experiment 3 (terrain robustness).

Produces reproducible MuJoCo scene XMLs:
  - Slopes: a flat run-up, then an inclined plane at {5,10,15,20,25} deg
  - Stairs: a flat run-up, then ascending steps of height {2,5,8,12,16} cm

All scenes reuse the self-contained robot at scenes/go2_model/go2.xml.
Scenes are written to scenes/ and are committed to the repo so the exact
terrain used in the paper is reproducible.

Run once:  python3 generate_terrain_scenes.py
"""

import os
import numpy as np

SCENES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "scenes")
)

# Shared header/footer ------------------------------------------------------
HEADER = """<mujoco model="{model_name}">
  <include file="go2_model/go2.xml"/>
  <statistic center="0 0 0.1" extent="0.8"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-130" elevation="-20"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
             width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
             markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
              texrepeat="5 5" reflectance="0.2"/>
    <material name="rampmat" rgba="0.45 0.45 0.55 1"/>
    <material name="stepmat" rgba="0.5 0.42 0.38 1"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
"""

FOOTER = """  </worldbody>
</mujoco>
"""


def make_slope_scene(angle_deg):
    """
    Flat run-up (so the policy settles before the slope), then a long
    inclined plane tilted by angle_deg about the Y axis, rising toward +X.

    The robot starts at the origin on flat ground and walks toward +X.
    """
    a = np.radians(angle_deg)

    # Flat run-up: a box from x in [-1, 1.5], top surface at z=0
    flat_len = 2.5
    flat_cx = 0.25  # center x of run-up box covering x in [-1, 1.5]

    # Ramp: a thin tilted box. Length 8 m along its surface.
    ramp_len = 8.0
    ramp_thick = 0.1
    # The ramp begins where the flat ends (x = 1.5).
    x_start = 1.5
    # Center of the ramp surface along world X and Z:
    # surface midpoint is at x_start + (ramp_len/2)*cos(a), z = (ramp_len/2)*sin(a)
    cx = x_start + (ramp_len / 2.0) * np.cos(a)
    cz = (ramp_len / 2.0) * np.sin(a) - (ramp_thick / 2.0) * np.cos(a)

    body = f"""    <!-- Flat run-up -->
    <geom name="floor" type="plane" size="0 0 0.05" material="groundplane"/>
    <geom name="runup" type="box" pos="{flat_cx} 0 -0.05"
          size="{flat_len/2:.4f} 2.0 0.05" material="groundplane"/>
    <!-- Inclined plane: {angle_deg} deg -->
    <geom name="ramp" type="box"
          pos="{cx:.4f} 0 {cz:.4f}"
          size="{ramp_len/2:.4f} 2.0 {ramp_thick/2:.4f}"
          euler="0 {-a:.6f} 0"
          material="rampmat"/>
"""
    name = f"go2_slope_{angle_deg:02d}.xml"
    xml = HEADER.format(model_name=f"go2 slope {angle_deg}deg") + body + FOOTER
    return name, xml


def make_stairs_scene(step_h_cm):
    """
    Flat run-up then a staircase ascending toward +X.
    Step height = step_h_cm centimeters, tread (depth) = 30 cm, fixed.
    10 steps total.
    """
    step_h = step_h_cm / 100.0
    tread = 0.30
    n_steps = 10
    width = 2.0

    flat_len = 2.5
    flat_cx = 0.25
    x_stair_start = 1.5

    body = [
        '    <!-- Flat run-up -->',
        '    <geom name="floor" type="plane" size="0 0 0.05" material="groundplane"/>',
        f'    <geom name="runup" type="box" pos="{flat_cx} 0 -0.05" '
        f'size="{flat_len/2:.4f} 2.0 0.05" material="groundplane"/>',
        f'    <!-- Staircase: {n_steps} steps, {step_h_cm} cm rise, '
        f'{tread*100:.0f} cm tread -->',
    ]

    for i in range(n_steps):
        # Step i: top surface at z = (i+1)*step_h
        # spans x in [x_stair_start + i*tread, x_stair_start + (i+1)*tread]
        top_z = (i + 1) * step_h
        x0 = x_stair_start + i * tread
        # Model each step as a box from z=0 up to top_z (solid riser+tread).
        cx = x0 + tread / 2.0
        cz = top_z / 2.0
        body.append(
            f'    <geom name="step_{i}" type="box" '
            f'pos="{cx:.4f} 0 {cz:.4f}" '
            f'size="{tread/2:.4f} {width/2:.4f} {top_z/2:.6f}" '
            f'material="stepmat"/>'
        )

    name = f"go2_stairs_{step_h_cm:02d}.xml"
    xml = (HEADER.format(model_name=f"go2 stairs {step_h_cm}cm")
           + "\n".join(body) + "\n" + FOOTER)
    return name, xml


def main():
    os.makedirs(SCENES_DIR, exist_ok=True)

    slope_angles = [5, 10, 15, 20, 25]
    step_heights_cm = [2, 5, 8, 12, 16]

    written = []

    for ang in slope_angles:
        name, xml = make_slope_scene(ang)
        path = os.path.join(SCENES_DIR, name)
        with open(path, "w") as f:
            f.write(xml)
        written.append(name)

    for h in step_heights_cm:
        name, xml = make_stairs_scene(h)
        path = os.path.join(SCENES_DIR, name)
        with open(path, "w") as f:
            f.write(xml)
        written.append(name)

    print(f"Wrote {len(written)} terrain scenes to {SCENES_DIR}:")
    for n in written:
        print(f"  {n}")


if __name__ == "__main__":
    main()