# Chapter 6 — From Toy to Robot: Continuous Control, Observation Design, and Sim-to-Real

## What this chapter is about

You've trained PPO on three toy environments. Now think about what changes when you go from those toys to a real robot like the Unitree Go2.

Conceptually, very little. The algorithm is identical. The observation-action-reward loop is identical. The Bellman equation, the policy gradient, the clipping — all unchanged.

But practically, *a lot* changes. The dimensionality explodes. Continuous control demands a different policy architecture. Observation design becomes an engineering problem in its own right. And — the hardest one — your trained policy has to work on a robot whose physics aren't exactly the simulation's. This chapter is the bridge between "I can train PPO on Pendulum" and "I can deploy a real RL controller on a quadruped."

---

## The dimensionality jump

The toy environments you trained were tiny:

| Env | Obs dim | Action dim | Action type |
|---|---|---|---|
| CartPole | 4 | 2 | Discrete |
| LunarLander | 8 | 4 | Discrete |
| Pendulum | 3 | 1 | Continuous |
| **Go2 (walk-these-ways)** | **70** | **12** | **Continuous** |

70 dimensions doesn't sound like a huge jump from 8, but the *content* of those 70 dimensions is qualitatively different. They include joint positions, joint velocities, body orientation (as projected gravity), commands from a user/operator, gait clock signals, and a history of recent actions. Each of those subgroups needs to be in a specific format, in a specific order, at a specific scale.

For the Go2, the action space is also more complex. It's not just "12 numbers" — it's 12 *coordinated* numbers that have to produce a stable gait. Each joint affects the body's center of mass, its angular momentum, the contact forces under each foot. The combinatorics get vicious fast.

This is why training a legged-robot policy is GPU-intensive: the environments are higher-dimensional, the rewards are more complex, and you need *thousands* of parallel envs to collect enough data to learn from in reasonable time. Stage 3 of this project is where we cross that bridge.

---

## Continuous control changes how policies output actions

You already met this in Pendulum. Just to anchor it:

For **discrete actions**, the policy network outputs a vector of probabilities — one per choice. You sample from a categorical distribution. CartPole: `softmax([logit_left, logit_right]) → [p_left, p_right]`, sample an action.

For **continuous actions**, the policy outputs the *parameters of a continuous distribution* — typically a Gaussian (normal distribution) per action dimension. So a 12-dim continuous policy outputs:

- 12 **means** (the "preferred" action value)
- 12 **standard deviations** (how much exploration noise to add)

Sampling: `action = mean + noise × std`, where noise is drawn from a unit Gaussian.

The std starts high (lots of exploration) and shrinks as training converges. That's what you watched in Pendulum's `entropy_loss` curve. By the end of training, std is small and the policy is essentially deterministic — it picks roughly the same action each time it sees a given state.

For deployment, you usually use the mean directly (no exploration noise), called the **deterministic policy** at inference time. That's why `model.predict(obs, deterministic=True)` was in your `02_cartpole_watch.py` and `04_lunarlander_watch.py` scripts.

---

## Observation design is its own discipline

In CartPole, the observation was given to you. 4 numbers, well-scaled, fully Markovian. Easy.

For a robot, *you design the observation*. Decisions you make include:

**What to include.** Joint positions, sure. Joint velocities, yes. Body orientation, yes. But also: should you include the body's velocity in world frame? In body frame? Should you tell the policy what commands the operator is sending? Should you include sensor readings from a depth camera?

Each addition increases the obs dimension but might or might not help learning. Too sparse → policy can't act. Too rich → learning becomes slow because the policy must learn what's relevant.

**What frame to express things in.** This is huge. World-frame velocity tells you "the robot is moving north at 0.5 m/s." Body-frame velocity tells you "the robot is moving forward at 0.5 m/s." For a legged robot that turns frequently, body-frame is far more useful — the policy reasons about "should I lift my front-right foot?" in body coordinates, not world coordinates. The walk-these-ways obs vector is almost entirely body-frame for this reason.

**How to scale things.** Joint angles in radians are typically in [−1.5, 1.5]. Joint velocities can hit ±25 rad/s. Linear velocities are in m/s. These have wildly different magnitudes, and a neural network's gradients will get dominated by the largest. You scale each piece into a similar range during training, and you must reproduce that exact scaling during deployment. That's what `OBS_SCALES` was in your Stage 2 code.

**How to encode time-dependent information.** A walking policy needs a sense of rhythm — when to lift each foot. The walk-these-ways policy includes 4-dimensional "clock signals" (sines of per-foot phases) as part of the obs. These are constructed externally, not measured. They give the policy a pacemaker. Without them, the policy wouldn't know when to step.

This is why we spent so much time in Stages 2.3 and 2.4 building the obs vector by hand. The format isn't optional — it's part of the policy itself.

---

## Sim-to-sim and sim-to-real

Training a policy in simulation and deploying it on a real robot is called **sim-to-real transfer**. It's the holy grail of modern robotics RL, and it's hard. There are many reasons:

- **Physics differences.** Real friction isn't constant. Real motors have backlash. Real sensors have noise. Real terrain isn't perfectly flat.
- **Latency.** Real motors take milliseconds to respond. Real sensors take milliseconds to read. The simulator is instantaneous.
- **Sensor noise.** Real IMUs drift. Real cameras have motion blur.
- **Unmodeled dynamics.** The robot has wires, payloads, mounted hardware that the simulation didn't account for.

The standard technique is **domain randomization**: during training, randomize physics parameters across simulated episodes — friction coefficients, motor strength, base mass, sensor noise, latency. The policy learns to be robust to a *range* of physical conditions, so when it encounters the specific conditions of the real robot, it's already prepared.

The walk-these-ways training config you saw earlier had domain randomization for: friction, restitution, base mass, motor strength, motor offset, gravity, control latency. Twelve different parameters were randomized per training episode. This is what made the policy transferable.

**Sim-to-sim** — what you did in Stage 2 — is a smaller version of the same problem. Even between simulators (Isaac Gym to MuJoCo), physics differs. Contact models differ. Solver tolerances differ. Foot friction differs. The same policy will behave subtly differently in each. You experienced this firsthand: the robot took your fixes to walk in MuJoCo, even though it walked beautifully in the Isaac Gym training videos. The sim-to-sim gap was small but non-zero.

---

## What you should remember

- The fundamental RL machinery doesn't change going from toys to robots. PPO is PPO.
- What changes is dimensionality, continuous action distributions, observation engineering, and the gap between training and deployment environments.
- Designing the observation vector is part of designing the policy. Get it wrong and the policy can't learn (during training) or won't work (during deployment).
- Sim-to-real is solved primarily by domain randomization, not by making the simulator more accurate. You make the policy robust enough that it doesn't *need* an accurate simulator.

---

## What we did in our project

The journey from Stage 1 to Stage 2 traced every concept in this chapter:

- **Dimensionality jump:** From 3 dims (Pendulum) to 70 dims (Go2). The same PPO works for both — you didn't change the algorithm, you changed the environment.
- **Continuous control:** Pendulum was your first taste of Gaussian policies. The walk-these-ways policy is the same idea at scale — 12 continuous outputs instead of 1.
- **Observation design:** You read the actual training source code to find out what those 70 dimensions are. You manually built the obs vector. You discovered scales, body-frame conversions, and clock signals along the way. That's observation engineering.
- **Sim-to-sim transfer:** Stage 2.5's debugging journey *was* sim-to-sim transfer in miniature. Wrong clock signal, wrong default pose, wrong hip signs — each one a small mismatch between training and deployment conditions. Fixing each was a sim-to-sim domain randomization patch applied by hand.
- **Domain randomization mention:** When we get to Stage 3 (training our own policy), domain randomization will become a configurable thing we explicitly tune. We'll randomize friction, mass, motor strength, and more.

Chapter 7 dives into the specific architecture and training tricks that walk-these-ways uses: how gait-conditioning works, why there's an "adaptation module," and what RMA (Rapid Motor Adaptation) actually does. By the end of it, every weird detail in your Stage 2 code will make perfect sense.
