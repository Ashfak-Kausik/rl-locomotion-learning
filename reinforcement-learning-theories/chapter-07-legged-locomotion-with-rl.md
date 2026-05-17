# Chapter 7 — Legged Locomotion with RL: walk-these-ways, RMA, and Gait Conditioning

## What this chapter is about

You've deployed the `walk-these-ways` policy. You know it's a neural network. You know it takes 70 observations in and outputs 12 actions. But what's actually inside that network architecture, and what design decisions made it work? Why two networks instead of one? Why a 30-step history? Why a 4-dimensional clock signal?

This chapter answers all of those questions. We're going to unpack three ideas that define modern legged-robot RL:

1. **Rapid Motor Adaptation (RMA)** — the two-network trick that lets a single policy adapt to environments it wasn't directly trained on.
2. **Gait conditioning** — how walk-these-ways lets a user command not just *velocity* but *gait style* (trot, pace, bound, etc.).
3. **Reward shaping for locomotion** — why the training reward has 20+ terms and what each one prevents.

After this chapter, every line of your Stage 2.5 code will have a *why* attached to it.

---

## The blind locomotion problem

Before we get to RMA, understand what makes legged locomotion hard.

A quadruped policy has to handle environments the robot has never seen. The training simulator can randomize friction, mass, terrain, motor strength — but at deployment, the robot meets *specific* values of those parameters. A policy trained on randomized environments must produce a single set of joint commands that works *across* the entire randomized range. That's a tall order.

The intuitive fix is to add sensors: friction sensors on the feet, force-torque sensors on the joints, etc. But these sensors are expensive, fragile, and add latency. Most legged robots, including the Go2, ship without them. So the policy has to work with **proprioception only** — joint angles, joint velocities, body orientation, IMU. No external sensing of the environment.

This is called *blind locomotion*, and it's surprisingly powerful. A blind policy can climb stairs, walk on grass, recover from slips, and traverse rubble — all while only "feeling" its own joints. The policy has to *infer* the environment from how the body responds to its own commands.

That inference is what RMA mechanizes.

---

## Rapid Motor Adaptation (RMA)

RMA (Kumar et al., RSS 2021) is the architecture inside walk-these-ways. It's two networks working together:

**The body** — the main policy network. Takes the observation and outputs joint commands. This is what walks the robot.

**The adaptation module** — a smaller network that looks at the last 30 observations and outputs a 2-dimensional vector. That vector is the body's "environment estimate." It's not measuring anything directly; it's a learned summary of "based on how my body has been moving over the last 600 ms, what kind of world am I in?"

Here's the magic: the adaptation module's output is concatenated to the body's input. So the body doesn't just see the current state — it sees the current state *plus* the adaptation module's environmental guess.

During training, RMA actually uses a clever two-phase trick:

**Phase 1.** Train a privileged policy with access to ground-truth environment parameters (friction, mass, motor strength, etc.). This policy works perfectly because it knows the environment exactly. Call its environment representation `z_true`.

**Phase 2.** Train the adaptation module to predict `z_true` from observation history alone. Replace the privileged environmental info with the adaptation module's output. The body keeps working because the adaptation module learned to recover environmental information from how the body has been moving.

Effectively, the adaptation module is a **learned state estimator** for hidden environment properties. It's the policy's substitute for friction sensors and force gauges.

When you concatenated `history_tensor` (2100 dims) with `env_latent` (2 dims) in your Stage 2.5 code to get a 2102-dim body input, you were running this exact architecture.

---

## Gait conditioning — `walk-these-ways`'s contribution

If RMA was the framework, the gait-conditioning idea is what made `walk-these-ways` (Margolis & Agrawal, CoRL 2023) distinct. The earlier RMA paper trained a single policy that walked one way. `walk-these-ways` trains a single policy that can be commanded to walk *many ways*.

Here's how. Recall your 15-dim command vector:
[lin_vel_x, lin_vel_y, ang_vel_yaw,
body_height,
step_frequency, gait_phase, gait_offset, gait_bound, gait_duration,
footswing_height,
body_pitch, body_roll,
stance_width, stance_length,
Aux_reward]

The first three are velocity commands — what direction to go and how fast to turn. The rest are **gait shape commands**.

By varying `gait_phase`, `gait_offset`, `gait_bound`, you can produce different gaits:

- `gait_phase=0.5, offset=0, bound=0` → **trot** (diagonal pairs together — the gait you commanded)
- `gait_phase=0, offset=0, bound=0` → **pronk/bound** (all four feet together — what you accidentally got when the clock signal was broken)
- `gait_phase=0, offset=0.5, bound=0` → **pace** (left pair, then right pair)
- `gait_phase=0.5, offset=0, bound=0.5` → various combinations

During training, the policy is conditioned on these commands and learns to actually produce the commanded gait. The policy is essentially solving 100 gait problems at once, with the gait commands telling it which one to solve right now.

The clock signals — those 4 sines you computed — encode *where in the gait cycle* each foot should currently be. The policy was trained to match its actual foot motion to these clock signals. That's why getting the clock signal right was crucial in your debugging: with the wrong clocks, the policy didn't know when to step.

**Why gait conditioning matters beyond style.** Different gaits suit different terrains. Trot is efficient on flat ground. Pace is efficient in narrow corridors. Bound is faster but less stable. A single policy that can do all of them lets the operator (or a higher-level controller) match the gait to the situation. This is much more powerful than a policy locked to one walking style.

---

## Reward shaping for legged locomotion

You saw the training config dump. The reward had 20+ components. Why?

A legged robot can fail in many ways:
- Fall over (body height too low)
- Wobble or oscillate (instability)
- Walk slowly when commanded to walk fast (poor velocity tracking)
- Drift sideways when commanded forward (poor heading)
- Slide its feet on the ground (poor contact discipline)
- Drag its body (low body height)
- Lift its feet too high (wasted energy)
- Stomp violently (high impact forces)
- Bounce in place (no actual progress)
- Move with the wrong gait
- Apply too much torque to one joint (motor damage)
- Take steps of inconsistent length

Each one of these has a reward term aimed at it. Looking at your training config:

| Reward term | Sign | What it incentivizes |
|---|---|---|
| `tracking_lin_vel` | + | Matching commanded forward/lateral velocity |
| `tracking_ang_vel` | + | Matching commanded turn rate |
| `lin_vel_z` | − | Punishes vertical bouncing |
| `ang_vel_xy` | − | Punishes pitching/rolling |
| `orientation_control` | − | Punishes orientation deviation from command |
| `torques` | − | Punishes high torque (energy + motor wear) |
| `dof_acc` | − | Punishes jerky joint motion |
| `action_rate` | − | Punishes rapidly changing commands |
| `action_smoothness_1/2` | − | Higher-order smoothness terms |
| `collision` | − | Punishes leg-body collisions |
| `feet_slip` | − | Punishes feet sliding when in stance |
| `feet_clearance_cmd_linear` | − | Punishes wrong footswing height |
| `tracking_contacts_shaped_force/vel` | + | Rewards matching the commanded gait timing |
| `jump` | + | Rewards intentional jumps when commanded |
| `raibert_heuristic` | − | Encourages foot placement that maintains balance |
| `dof_pos_limits` | − | Punishes hitting joint limits |
| `survival` | + | Bonus for not falling |
| `feet_contact_forces` | (off) | Could punish too-hard contacts |
| `energy` / `energy_expenditure` | (off) | Could punish wasted energy |

The signs and magnitudes were tuned over hundreds of training runs. Many of those components are zero in the default walk-these-ways config but available to enable. Each one represents a failure mode the researchers saw and patched.

**The lesson for reward design:** start simple, see what fails, add a term to fix the failure, repeat. You're not writing one reward; you're sculpting a behavior by sequential constraint.

---

## The training pipeline (preview)

When you train a policy yourself in Stage 3, here's roughly what will happen:

1. Initialize 4096 simulated environments in parallel (on GPU).
2. Initialize the body and adaptation module with random weights.
3. Each iteration:
   - Roll out 24 timesteps in each env.
   - Compute returns, advantages.
   - Run PPO updates on the body.
   - Update the adaptation module (predicting privileged environmental info).
4. Apply domain randomization: every few seconds, resample friction, mass, motor strength.
5. After 1500 iterations (~3-6 hours on a good GPU), the policy is ready.

The output is `body_latest.jit` and `adaptation_module_latest.jit` — the same kinds of files you loaded in Stage 2.5, but trained by you.

---

## What you should remember

- Blind locomotion is the dominant paradigm; the policy "feels" the environment through how its body responds, not via dedicated sensors.
- RMA's two-network structure lets a policy adapt to unseen environments by having a learned state estimator (the adaptation module) feed environmental guesses to the body.
- Gait conditioning lets one policy do many gaits by including gait shape commands in the observation; the clock signals encode the rhythm.
- Reward design for legged robots is sculpting behavior with many small constraints, each preventing a specific failure mode.

---

## What we did in our project

Stage 2 was unwitting use of every concept here:

- **Blind locomotion:** Our policy uses only proprioception. No cameras, no force sensors. We literally couldn't add them even if we wanted to — the policy wasn't trained with vision.
- **RMA in action:** The `adaptation_module_latest.jit` is the adaptation module. The `body_latest.jit` is the body. We concatenated their outputs into a 2102-dim body input. RMA, deployed.
- **Gait conditioning:** Our `COMMANDS` array sets `gait_phase=0.5, gait_offset=0, gait_bound=0` — that's a trot. Setting `gait_phase=0` would have given us bound. We could have set `body_height=0.05` to make the robot crouch while walking. All without retraining.
- **The clock signal saga:** Our biggest debugging insight was that the policy uses 4 per-foot sines (not sin/cos pairs) and that the foot phases are constructed via specific offsets of the gait command parameters. We rebuilt the exact training code's phase logic in our deployment script.
- **The complexity of reward (preview for Stage 3):** When we train our own policy, we'll start with a smaller set of reward terms and gradually add more as we see failure modes. This is the iteration loop researchers actually use.

The next chapter — Chapter 8 — pulls every thread of this project together. Every script, every result, every debugging insight, every TensorBoard curve. It's the long one. The deep dive you've been building toward.