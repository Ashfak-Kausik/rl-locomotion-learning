# Tech Stack Primer

Everything you need to know about the languages, libraries and APIs used in
this repo — taught through code that is *actually in it*. Written for someone
who can program but has not used Python-for-robotics, MuJoCo, or RL libraries
before.

**Read this alongside [ARCHITECTURE.md](ARCHITECTURE.md).** Architecture says
*what connects to what*; this says *what the syntax means*.

---

## Contents

1. [The stack at a glance](#1-the-stack-at-a-glance)
2. [Python, the parts this repo leans on](#2-python-the-parts-this-repo-leans-on)
3. [NumPy — the array language](#3-numpy--the-array-language)
4. [MuJoCo — the physics engine](#4-mujoco--the-physics-engine)
5. [MJCF — the robot description format](#5-mjcf--the-robot-description-format)
6. [PyTorch & TorchScript — inference only](#6-pytorch--torchscript--inference-only)
7. [Gymnasium — the RL environment API](#7-gymnasium--the-rl-environment-api)
8. [Stable-Baselines3 — PPO without writing PPO](#8-stable-baselines3--ppo-without-writing-ppo)
9. [TensorBoard — reading training curves](#9-tensorboard--reading-training-curves)
10. [Matplotlib — the figure pipeline](#10-matplotlib--the-figure-pipeline)
11. [The robotics concepts you cannot skip](#11-the-robotics-concepts-you-cannot-skip)
12. [Gotchas that will cost you an afternoon](#12-gotchas-that-will-cost-you-an-afternoon)

---

## 1. The stack at a glance

| Layer | Tool | Used for | Where |
|---|---|---|---|
| Language | Python 3.10+ | everything | all |
| Arrays | NumPy | state vectors, metrics | all |
| Physics | MuJoCo 3.x | rigid-body simulation | Stage 2 |
| Robot format | MJCF (XML) | robot + world description | `scenes/` |
| NN runtime | PyTorch (TorchScript) | policy **inference** | Stage 2 |
| RL envs | Gymnasium | CartPole/LunarLander/Pendulum | Stage 1 |
| RL algos | Stable-Baselines3 | PPO **training** | Stage 1 |
| Diagnostics | TensorBoard | training curves | Stage 1 |
| Figures | Matplotlib | paper plots | Stage 2 |
| Reproducibility | host venv + `requirements.txt` | pinned environment | repo-wide |

**Note the asymmetry**: Stage 1 *trains* neural networks with a high-level
library. Stage 2 only *runs* an already-trained one, but hand-writes
everything around it. Different skills entirely.

---

## 2. Python, the parts this repo leans on

You do not need advanced Python. You need these six things fluently.

### Slicing

The single most-used operation in the codebase.

```python
data.qpos[7:19]     # elements 7,8,...,18   → 12 joint angles
data.qvel[6:18]     # elements 6,7,...,17   → 12 joint velocities
data.qpos[0:3]      # base position (x, y, z)
data.qpos[3:7]      # base orientation quaternion (w, x, y, z)
```

`[a:b]` is inclusive of `a`, exclusive of `b`. Everything in Stage 2 is
slicing a flat physics-state array into meaningful chunks.

### f-strings and format specs

```python
print(f"Time: {data.time:.2f}s | height={base_height:.3f}m")
#                          ^^^^ 2 decimal places
print(f"{surv_rate:>7.0%}")   # right-align width 7, render as a percentage
```

### Dictionaries as config

```python
OBS_SCALES = {"lin_vel": 2.0, "ang_vel": 0.25, "dof_vel": 0.05}
value = OBS_SCALES["dof_vel"]        # KeyError if the key is absent — good,
                                     # a typo fails loudly instead of silently
```

### `collections.deque` — the history buffer

```python
from collections import deque

obs_history = deque([np.zeros(70) for _ in range(30)], maxlen=30)
obs_history.append(obs)     # oldest element is dropped automatically
```

`maxlen` makes this a fixed-size sliding window. Appending to a full deque
evicts from the other end in O(1). This is the robot's 0.6 s memory.

### Context managers (`with`)

```python
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        ...
# viewer window is guaranteed closed here, even if the loop raised
```

### The `if __name__ == "__main__":` guard

```python
def run_trial(...): ...

if __name__ == "__main__":
    # only runs when the file is executed directly, not when imported
    print(json.dumps(run_trial(scene), indent=2))
```

This is why `harness.py` can be both an importable library (used by
`exp1/2/3`) and a runnable smoke test.

---

## 3. NumPy — the array language

NumPy arrays are fixed-type, fixed-size, and support element-wise maths
without loops. Physics state and neural network I/O are both just arrays.

```python
import numpy as np

# --- creation
np.zeros(12)                      # 12 zeros
np.array([0.1, 0.8, -1.5])        # from a list
np.full(12, 0.25)                 # 12 copies of 0.25

# --- element-wise arithmetic (no loop!)
joint_pos_obs = (joint_pos - DEFAULT_JOINT_POS) * OBS_SCALES["dof_pos"]
#                └─ 12 subtractions ─┘          └─ 12 multiplications ─┘

# --- concatenation: how the 70-dim obs is assembled
obs = np.concatenate([proj_grav, cmd, joint_pos_obs, joint_vel_obs,
                      last_action, prev_action, clock])
#                        3   +  15  +  12  +  12  +  12  +  12  + 4 = 70

# --- shape assertions catch layout bugs immediately
assert obs.shape == (70,), f"obs shape is {obs.shape}, expected (70,)"

# --- linear algebra used here
np.cross(a, b)                    # 3-D cross product (quaternion rotation)
a @ b                             # dot product
np.linalg.norm(end[:2] - start[:2])   # Euclidean distance travelled

# --- statistics for the experiment metrics
vx = np.array(vx_log)
vx.mean(), vx.std()
np.abs(vy_arr).mean()             # mean magnitude, ignoring direction

# --- trig, vectorised over all 4 feet at once
clock = np.sin(2 * np.pi * foot_phases)   # foot_phases is shape (4,)

# --- modulo wraps the gait phase into [0, 1)
foot_phases = np.array([...]) % 1.0

# --- seeded randomness: same seed ⇒ same trial, every time
rng = np.random.default_rng(seed)
data.qpos[7:19] += rng.uniform(-0.02, 0.02, size=12)
```

> **`.copy()` is not optional.** NumPy slices are *views* into the same
> memory. `standing_target = data.qpos[7:19]` would keep changing as the
> simulation runs; `data.qpos[7:19].copy()` freezes it. `03_pose_go2.py`
> calls this out explicitly, and it is a real bug source.

---

## 4. MuJoCo — the physics engine

MuJoCo (Multi-Joint dynamics with Contact) is a rigid-body simulator built for
robotics. Two objects, and the distinction is everything:

```python
import mujoco

model = mujoco.MjModel.from_xml_path(ABSOLUTE_PATH)   # STATIC  — never changes
data  = mujoco.MjData(model)                          # DYNAMIC — changes every step
```

| `model` (MjModel) | `data` (MjData) |
|---|---|
| masses, geometry, joint limits | positions, velocities, forces |
| `nq`, `nv`, `nu` (state sizes) | `qpos`, `qvel`, `ctrl`, `time` |
| loaded once from XML | mutated 500×/second |

### The core API, complete

```python
# --- stepping
mujoco.mj_step(model, data)       # advance one timestep (integrate dynamics)
mujoco.mj_forward(model, data)    # recompute derived quantities WITHOUT
                                  # advancing time — call after you poke qpos

# --- named lookup: names live in the XML, indices live in the arrays
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(model, data, key_id)

name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
#   mjOBJ_BODY / mjOBJ_JOINT / mjOBJ_ACTUATOR / mjOBJ_SENSOR / mjOBJ_KEY

# --- the state you read and write
data.qpos      # (nq,) generalised positions   — 19 for the Go2
data.qvel      # (nv,) generalised velocities  — 18 for the Go2
data.ctrl      # (nu,) actuator commands       — 12 for the Go2
data.time      # simulated seconds

model.opt.timestep    # 0.002 s → 500 Hz
model.nu, model.nq, model.nv, model.nbody, model.njnt
```

### Why `nq` (19) ≠ `nv` (18)

The Go2's floating base is a *free joint*. Its orientation needs **4** numbers
in position space (a quaternion) but only **3** in velocity space (an angular
velocity vector). So `qpos` has one extra slot. This is why joint angles start
at `qpos[7]` but joint velocities start at `qvel[6]`.

### Viewer vs. headless

```python
# Interactive window — real-time pacing, human in the loop
import mujoco.viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()

# "passive" = YOU drive the loop; the viewer only draws.
# There is also launch() which owns the loop — this repo never uses it,
# because the control code must run between steps.

# Headless offscreen rendering — for screenshots/video with no window
#
# GOTCHA: the offscreen framebuffer defaults to 640x480, and Renderer REFUSES
# a larger request rather than resizing. Enlarge it on the MODEL first, before
# constructing any Renderer:
model.vis.global_.offwidth = 1920
model.vis.global_.offheight = 1080

renderer = mujoco.Renderer(model, height=1080, width=1920)
renderer.update_scene(data)
frame = renderer.render()          # a (H, W, 3) uint8 NumPy array
renderer.close()

# Equivalently, declare it in the scene XML:
#   <visual><global offwidth="1920" offheight="1080"/></visual>

# No rendering at all — what the experiments do. Fastest by far.
for step in range(n_steps):
    mujoco.mj_step(model, data)
```

The `MUJOCO_GL` environment variable picks the rendering backend:

| value | meaning | when |
|---|---|---|
| `glfw` (default) | on-screen window | desktop with a display |
| `osmesa` | CPU software rendering | headless servers, no GPU |
| `egl` | GPU rendering, no window | headless machine with a GPU |

---

## 5. MJCF — the robot description format

MJCF is MuJoCo's XML dialect. `scenes/go2_flat.xml` is a complete, readable
example:

```xml
<mujoco model="go2 flat scene">
  <include file="go2_model/go2.xml"/>          <!-- pull in the robot -->

  <asset>                                       <!-- textures & materials -->
    <texture type="2d" name="groundplane" builtin="checker" .../>
    <material name="groundplane" texture="groundplane" .../>
  </asset>

  <worldbody>                                   <!-- the physical scene -->
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>
</mujoco>
```

The robot file, `scenes/go2_model/go2.xml`, adds the parts that matter for control:

```xml
<compiler angle="radian" meshdir="." autolimits="true"/>
<!-- angle="radian": every angle in this file is radians, not degrees -->

<default>
  <default class="go2">
    <geom friction="0.6" margin="0.001" condim="1"/>
    <joint damping="2" armature="0.01" frictionloss="0.2"/>
    <motor ctrlrange="-23.7 23.7"/>              <!-- TORQUE limits, in Nm -->
    <default class="knee">
      <motor ctrlrange="-45.43 45.43"/>          <!-- knees are stronger -->
    </default>
  </default>
</default>

<actuator>
  <motor class="abduction" name="FL_hip"   joint="FL_hip_joint"/>
  <motor class="hip"       name="FL_thigh" joint="FL_thigh_joint"/>
  <motor class="knee"      name="FL_calf"  joint="FL_calf_joint"/>
  ...  <!-- 12 total: FL, FR, RL, RR × (hip, thigh, calf) -->
</actuator>

<keyframe>
  <key name="home" qpos="0 0 0.27  1 0 0 0  0 0.9 -1.8  ..."/>
</keyframe>
```

**The critical line is `<motor>`.** These are *torque* actuators, so
`data.ctrl[i]` is a torque in newton-metres — **not** a target angle. That is
the entire reason this repo hand-writes a PD controller. If the XML had used
`<position>` actuators instead, `data.ctrl` would take angles directly and the
PD loop would be unnecessary.

Terrain scenes are generated, not hand-written — `generate_terrain_scenes.py`
emits boxes with computed positions:

```xml
<geom name="ramp" type="box" pos="5.36 0 1.62"
      size="4.0 2.0 0.05" euler="0 -0.261799 0" material="rampmat"/>
<!-- euler is in radians here because of compiler angle="radian" -->
```

---

## 6. PyTorch & TorchScript — inference only

Stage 2 never trains anything. It loads two frozen graphs and calls them.

```python
import torch

# --- load a serialised model. NO class definition needed — TorchScript
#     bundles the architecture with the weights.
body_net  = torch.jit.load("body_latest.jit")
adapt_net = torch.jit.load("adaptation_module_latest.jit")

body_net.eval()          # inference mode (disables dropout/batchnorm training
                         # behaviour). Always call it.

# --- NumPy → tensor
history_tensor = torch.tensor(
    np.concatenate(obs_history).reshape(1, -1),   # (1, 2100)
    dtype=torch.float32,                          # networks want float32
)
#  reshape(1, -1): "1 row, infer the column count" → the batch dimension.
#  Networks expect batched input even for a single sample.

# --- inference
with torch.no_grad():                  # skip gradient bookkeeping:
    latent = adapt_net(history_tensor) #   ~2× faster, much less memory
    both   = torch.cat([history_tensor, latent], dim=1)   # (1, 2102)
    action = body_net(both).numpy().flatten()             # (12,)
#            tensor → NumPy ──┘        └── (1,12) → (12,)
```

That is the entire PyTorch surface area in this repo. Six functions.

**Why TorchScript matters here:** the policy was trained inside the
`walk-these-ways` Isaac Gym codebase, which is not installed here and never
will be. A `.pt` state-dict would need that codebase's model classes to load.
A `.jit` file is self-contained. This is the mechanism that makes sim-to-sim
transfer possible at all.

---

## 7. Gymnasium — the RL environment API

Gymnasium (the maintained fork of OpenAI Gym) standardises "what an RL
environment looks like". Learn these five calls and you can use any of the
thousands of environments that implement it.

```python
import gymnasium as gym

env = gym.make("CartPole-v1")              # or render_mode="human" to watch

obs, info = env.reset(seed=0)              # start an episode
#  ^^^ note: returns a TUPLE. Old Gym returned just obs — a very common
#      source of copy-pasted-code breakage.

obs, reward, terminated, truncated, info = env.step(action)
#                        ^^^^^^^^^^  ^^^^^^^^^
#    terminated = the episode genuinely ended (pole fell, lander crashed)
#    truncated  = a time limit was hit; the task itself did not end
#    Treating them as one boolean is wrong and biases value estimates.

env.close()
```

The three environments used, and what makes each different:

```
  CartPole-v1        obs (4,)  float  │  action: Discrete(2)
                     reward +1 per surviving step
                     terminates when the pole tips past ~12°

  LunarLander-v3     obs (8,)  float  │  action: Discrete(4)
                     shaped reward: distance + fuel + leg contact + crash
                     needs Box2D (a compiled C++ extension)

  Pendulum-v1        obs (3,)  float  │  action: Box(-2.0, 2.0, (1,))
                     reward always NEGATIVE, best possible is 0
                     NEVER terminates — always exactly 200 steps
```

`Discrete(n)` means "pick one of n". `Box(low, high, shape)` means "output a
real-valued vector in this range" — which is what a 12-joint robot needs, and
why Pendulum is the bridge environment in the curriculum.

### Vectorised environments

Running many copies at once gives PPO more diverse data per update:

```python
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

env = DummyVecEnv([lambda: gym.make("LunarLander-v3") for _ in range(8)])
env = VecMonitor(env)     # records episode reward/length for TensorBoard
```

`DummyVecEnv` runs all 8 sequentially in one process (simple, no IPC
overhead — right for cheap envs). `SubprocVecEnv` would use real processes.

> The VecEnv API differs from the plain Gym API: `reset()` returns only `obs`,
> and `step()` returns a 4-tuple. Mixing them up is the classic Stage 1 bug.

---

## 8. Stable-Baselines3 — PPO without writing PPO

SB3 supplies tested implementations of standard RL algorithms.

```python
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

model = PPO(
    "MlpPolicy",          # a multi-layer perceptron actor+critic; use
                          # "CnnPolicy" for image observations
    env,
    n_steps=2048,         # steps collected per env before each update
                          #   → batch = n_steps × n_envs
    batch_size=64,        # minibatch size for the gradient steps
    n_epochs=10,          # passes over that batch per update
    gamma=0.99,           # discount: how much future reward is worth now
    gae_lambda=0.95,      # bias/variance knob for advantage estimation
    learning_rate=1e-3,
    ent_coef=0.0,         # entropy bonus — raise it to force exploration
    use_sde=True,         # State-Dependent Exploration; helps continuous control
    verbose=1,
    tensorboard_log="./tb_logs/pendulum/",
)

model.learn(total_timesteps=300_000, progress_bar=True)
model.save("pendulum_ppo")                      # writes pendulum_ppo.zip

model = PPO.load("pendulum_ppo")
mean_reward, std_reward = evaluate_policy(model, gym.make("Pendulum-v1"),
                                          n_eval_episodes=100)

action, _states = model.predict(obs, deterministic=True)
#                                    ^^^^^^^^^^^^^^^^^^
#   True  → always take the distribution's mean (evaluation)
#   False → sample (training/exploration)
```

Hyperparameters actually used in this repo, and why they differ:

| Param | CartPole | LunarLander | Pendulum | Reasoning |
|---|---|---|---|---|
| `n_envs` | 1 | 8 | 8 | harder tasks need more parallel data |
| `n_steps` | default 2048 | 1024 | 2048 | ×8 envs → 8192 / 16384 per update |
| `gamma` | default 0.99 | 0.999 | 0.99 | LunarLander rewards arrive late |
| `ent_coef` | 0.0 | 0.01 | 0.0 | discrete needs a nudge to explore |
| `learning_rate` | 3e-4 | 3e-4 | 1e-3 | 1e-3 was *slightly too high* — see below |
| `use_sde` | — | — | `True` | smoother exploration in continuous space |

---

## 9. TensorBoard — reading training curves

```bash
tensorboard --logdir stage1-rl-fundamentals/tb_logs
# then open http://localhost:6006
```

Each `model.learn()` call creates `PPO_1`, `PPO_2`, … under the log dir, so
runs overlay for comparison automatically.

The metrics, what they mean, and what "bad" looks like:

| Metric | Reading it | Healthy |
|---|---|---|
| `rollout/ep_rew_mean` | mean episode reward — **the** score | rising, then plateau |
| `rollout/ep_len_mean` | mean episode length | task-dependent; **flat 200 for Pendulum by design — ignore it there** |
| `train/explained_variance` | how well the critic predicts returns | → 1.0. A *drop* signals destabilisation |
| `train/value_loss` | critic error | falling. Still falling at the end ⇒ train longer |
| `train/entropy_loss` | policy randomness (negative) | rises toward 0 as the agent commits |
| `train/std` | **continuous only** — action spread | shrinks as confidence grows |
| `train/clip_fraction` | fraction of updates PPO clipped | **0.1–0.3**. >0.4 ⇒ learning rate too high |
| `train/approx_kl` | policy change per update | **< 0.02**. Higher ⇒ steps too aggressive |

The Pendulum run in this repo shows `clip_fraction ≈ 0.45` and
`approx_kl ≈ 0.08` — both roughly 4× the healthy range, diagnosing the
`learning_rate=1e-3` choice. It still converged, which says something useful
about PPO's robustness. The full analysis is in
`stage1-rl-fundamentals/README.md`.

**The two failure modes to recognise:**
- *Policy collapse* — reward peaks then degrades with more training
  (LunarLander at ~860k steps here). Fix: `EvalCallback` to checkpoint the best model.
- *Plateau* — reward stops improving and simply stays there (Pendulum past
  300k). Harmless, just wasted compute.

---

## 10. Matplotlib — the figure pipeline

```python
import matplotlib
matplotlib.use("Agg")      # MUST come before pyplot import.
                           # "Agg" = write files, never open a window.
                           # Without it, headless runs crash.
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "figure.dpi": 300,
                     "savefig.bbox": "tight"})

fig, ax = plt.subplots(figsize=(6, 5))
ax.errorbar(cmds, means, yerr=stds, marker="o", capsize=4)  # mean ± std
ax.plot([0, lim], [0, lim], "--", color="gray", label="Ideal")
ax.set_xlabel("Commanded forward velocity (m/s)")
ax.legend()
fig.savefig("fig1.png")
plt.close(fig)             # free the figure — matters in a loop
```

Multi-panel figures use tuple unpacking:

```python
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))   # 1 row, 2 columns
```

---

## 11. The robotics concepts you cannot skip

### Quaternions and body frame

Orientation is stored as a 4-number quaternion `(w, x, y, z)` rather than
Euler angles, because quaternions have no gimbal lock and interpolate cleanly.

The policy never sees the raw quaternion. It sees **projected gravity** —
which way "down" points *from the robot's own point of view*:

```python
def quat_rotate_inverse(q, v):
    """Rotate v by the INVERSE of q: world frame → body frame."""
    q_w, q_vec = q[0], q[1:]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (2.0 * q_w)
    c = q_vec * 2.0 * (q_vec @ v)
    return a - b + c

proj_grav = quat_rotate_inverse(data.qpos[3:7], np.array([0.0, 0.0, -1.0]))
# upright  → [ 0,  0, -1]
# nose-down→ [-x,  0, -z]
# on its side → [0, ±1, 0]
```

Why body frame? A locomotion policy must behave identically whether the robot
faces north or south. Expressing everything relative to the body makes the
policy *rotation-invariant* — it cannot learn a spurious dependence on compass
heading. Same reasoning applies to velocity commands.

### PD control

Turning a desired angle into a torque:

```python
tau = KP * (target_angle - current_angle) - KD * current_velocity
#     └────── spring: pulls toward target ──┘  └── damper: resists motion ──┘
```

- `KP` too low → the robot sags under its own weight
- `KP` too high → oscillation and instability
- `KD` damps that oscillation; too high makes the joint sluggish

This repo uses `KP=25, KD=0.6` for policy-driven walking (matching the training
config), but `KP=100, KD=2.0` in `03_pose_go2.py` for simply standing still —
holding a static pose tolerates, and benefits from, much stiffer gains.

### Decimation

A control loop that runs slower than the physics loop. See
[ARCHITECTURE.md §6](ARCHITECTURE.md#6-the-dual-rate-control-loop). The policy
was *trained* at 50 Hz; running it at any other rate feeds it a world with the
wrong apparent dynamics.

### Sim-to-sim vs sim-to-real

- **sim-to-sim** — Isaac Gym → MuJoCo. Different contact solvers, integrators,
  and friction models. This is what the repo measures.
- **sim-to-real** — simulator → physical robot. Adds sensor noise, latency,
  motor backlash, battery sag. Stage 6, hardware permitting.

Sim-to-sim is the honest dress rehearsal: if a policy will not survive a
change of *simulator*, it will not survive reality.

---

## 12. Gotchas that will cost you an afternoon

| Symptom | Cause | Fix |
|---|---|---|
| `Error opening file '.../base_0.obj'` | relative path passed to `from_xml_path` | always `os.path.abspath(...)` — every script here already does |
| Robot immediately collapses | forgot `data.qpos[7:19] = DEFAULT_JOINT_POS` after the keyframe reset | the MJCF `home` pose ≠ the training pose |
| Robot twitches, never walks | wrong `DECIMATION`, so policy runs at the wrong Hz | 500 Hz physics ÷ 10 = 50 Hz policy |
| Legs splay outward | hip sign convention flipped | FL/RL `+0.1`, FR/RR `−0.1` |
| `ValueError: ... does not exist` on `.jit` | policy weights absent | `export GO2_POLICY_DIR=...` — see [DEPENDENCIES.md](DEPENDENCIES.md) |
| Matplotlib hangs/crashes headless | interactive backend | `matplotlib.use("Agg")` before importing pyplot |
| `mujoco.viewer` fails (no display / GLX) | headless host or driver mismatch | `MUJOCO_GL=egl` or `osmesa`; see NATIVE-SETUP.md |
| `Image width 900 > framebuffer width 640` | offscreen framebuffer defaults to 640×480 and `Renderer` will not resize | set `model.vis.global_.offwidth/offheight` **before** building the `Renderer` |
| `GLFWError: GLX: Failed to create context` on a desktop | GPU driver/library mismatch (often an NVIDIA update without a reboot) | reboot; or force software GL: `LIBGL_ALWAYS_SOFTWARE=1 __GLX_VENDOR_LIBRARY_NAME=mesa` |
| `Box2D` import error | Box2D not built | install `swig`, `build-essential`, `python3-dev`, then reinstall `gymnasium[box2d]` |
| Target array mutates on its own | NumPy slice is a view | `.copy()` |
| `obs, info = env.reset()` unpack error | mixing Gym / VecEnv APIs | VecEnv `reset()` returns obs only |
| SB3 `progress_bar=True` errors | missing deps | `pip install tqdm rich` |
| `externally-managed-environment` from pip | PEP 668 on Debian/Ubuntu | use the venv: `./scripts/setup_env.sh` |

---

## Where to go next

- **Theory** — `../reinforcement-learning-theories/` — 8 chapters written for
  this project, from MDPs through PPO to gait-conditioned locomotion.
- **Architecture** — [ARCHITECTURE.md](ARCHITECTURE.md)
- **Official docs** — [MuJoCo](https://mujoco.readthedocs.io) ·
  [Gymnasium](https://gymnasium.farama.org) ·
  [Stable-Baselines3](https://stable-baselines3.readthedocs.io) ·
  [PyTorch](https://pytorch.org/docs)
- **Source papers** — `walk-these-ways` (Margolis & Agrawal, CoRL 2022) ·
  RMA (Kumar et al., RSS 2021) · PPO (Schulman et al., 2017)
