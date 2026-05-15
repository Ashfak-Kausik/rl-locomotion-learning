# Chapter 2 — Core Concepts: States, Actions, Rewards, Policies, Values

## What this chapter is about

Chapter 1 gave you the five components of any RL system. Now we go deeper into each one. By the end of this chapter, you'll know the vocabulary, the symbols, and the subtle design choices that decide whether your RL setup works or fails. This is where casual readers and serious practitioners diverge — and you're about to join the second group.

We'll add two new concepts that weren't in Chapter 1: **value functions** and **trajectories**. They're how agents reason about the future, and you can't read a single RL paper without them.

---

## State (or observation)

The state is the snapshot of the world the agent uses to make its next decision. Two important distinctions to get out of the way:

**State vs. observation.** Strictly, the *state* is everything that fully describes the world — all the relevant variables. The *observation* is what the agent actually gets to see, which is often a subset. A chess engine sees the whole board (state = observation). A robot with a camera sees only what the camera frames (observation ⊂ state). In most modern RL we use the words interchangeably, but in theoretical papers the distinction matters.

**Markov property.** A state is *Markovian* if it contains enough information that you don't need any history to make a good decision. The position of every chess piece is Markovian — you don't need to know the move order to play well. A single camera frame of a moving car is *not* Markovian — you can't tell whether the car is speeding up or slowing down without prior frames. This matters because most RL algorithms assume Markovian states. When they're not, you typically stack several recent observations to *approximate* a Markovian state. (This is exactly why our walk-these-ways policy uses a 30-frame observation history — to give the adaptation module enough temporal context to estimate hidden environmental properties.)

We write the state at time `t` as **s_t**.

---

## Action

The action is what the agent does in response to a state.

Actions come in two flavors:

- **Discrete actions** — finitely many choices. Move up/down/left/right. Buy/sell/hold. Press button A, B, or C. CartPole and LunarLander have discrete actions.
- **Continuous actions** — real-valued numbers, often vectors. Apply 1.73 Nm of torque. Set the steering angle to 0.21 rad. Send 12 joint position targets. Pendulum and quadruped control have continuous actions.

The distinction matters because the policy architecture differs. Discrete policies output *probabilities over choices*. Continuous policies output *parameters of a probability distribution* (typically a Gaussian — a mean and a standard deviation per action dimension) from which you sample.

We write the action at time `t` as **a_t**.

---

## Reward

The reward is the agent's only signal of "how am I doing." It's a single number per time step.

Two principles every RL practitioner internalizes the hard way:

**Rewards must reflect what you actually want.** This sounds obvious but is the #1 cause of broken RL. If you reward a robot for "moving forward," it might learn to lurch forward and fall — which technically maximizes forward velocity until impact. If you reward it for "staying upright," it might learn to stand frozen forever. The reward shapes everything. Designing one that captures *all* the things you care about — speed, balance, smoothness, energy efficiency — without unintended exploits is real engineering. The `walk-these-ways` reward has 20+ terms exactly for this reason.

**Sparse vs. dense rewards.** A *sparse* reward gives feedback rarely (+100 only when the robot reaches the goal). A *dense* reward gives feedback every step (+0.1 per centimeter forward, −0.01 per joule of energy). Sparse rewards are honest but learning is slow because most steps give no signal. Dense rewards are easier to learn from but risk being gamed by the agent in ways you didn't anticipate. Most modern systems use dense rewards with careful design.

We write the reward at time `t+1` (received after taking action `a_t` in state `s_t`) as **R_{t+1}**.

---

## Trajectory

A *trajectory* (also called an *episode* or *rollout*) is the sequence of states, actions, and rewards the agent experiences:

$$\tau = (s_0, a_0, R_1, s_1, a_1, R_2, s_2, a_2, R_3, \ldots)$$

That τ (tau) is just a letter. It stands for a single "run" of the agent through the environment, from start to terminal state.

Trajectories are what RL algorithms learn from. The agent collects many trajectories (sometimes thousands per training iteration, in parallel), and uses them to update the policy. In your Stage 1 PPO runs, `n_steps=2048` meant "collect 2048 timesteps across all environments, then update the policy." That's a batch of partial trajectories.

---

## Return

We met this in Chapter 1 — the total discounted reward from time `t` onward:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

The return is what we actually want to maximize. The reward at any single step matters only because it contributes to the return.

A worked tiny example: suppose γ = 0.9 and the agent earns rewards [1, 2, 3, 4]:

$$G_0 = 1 + 0.9 \times 2 + 0.81 \times 3 + 0.729 \times 4 = 1 + 1.8 + 2.43 + 2.916 = 8.146$$

Notice how the later rewards get squeezed by the powers of γ. A reward 10 steps away is worth `0.9^10 ≈ 0.35` of itself today. That's the discount in action.

---

## Policy

A policy is the agent's strategy: a function that maps states to actions.

We write it as **π** (Greek "pi"). For a discrete policy: π(a | s) = the probability of choosing action `a` given state `s`. For a continuous policy: π is a function that outputs the parameters of an action distribution given `s`.

Crucially, **a policy in modern RL is almost always a neural network**. The network's weights *are* the policy. Training the policy means adjusting those weights so the policy collects higher returns on average.

In Stage 1 of this project, our policy was an MLP (multi-layer perceptron) — three hidden layers with ELU activations. In Stage 2, the `walk-these-ways` policy is similar but bigger: a 2102 → 512 → 256 → 128 → 12 architecture.

---

## Value functions

This is where many beginners stumble, so we'll go carefully.

A **value function** answers the question: *"if I'm in this state and follow my policy from now on, what return should I expect?"*

There are two common variants:

**State-value function**, written `V(s)`:

$$V^{\pi}(s) = \mathbb{E}_{\pi}\left[ G_t \mid s_t = s \right]$$

That `E_π[...]` is just "the expected value, assuming actions are sampled from policy π." In words: V(s) is the *average* return you'd get from state `s` if you played out many episodes under policy π.

**Action-value function**, written `Q(s, a)`:

$$Q^{\pi}(s, a) = \mathbb{E}_{\pi}\left[ G_t \mid s_t = s, a_t = a \right]$$

Q(s, a) is the average return from taking action `a` in state `s`, then following π afterward. It's strictly more informative than V — Q tells you per-action value.

**Why we care.** If you knew Q exactly, you'd just pick the action with the highest Q value in every state. Done. Optimal play. So one entire family of RL algorithms (Q-learning, DQN) tries to learn Q directly. Another family (policy gradients, PPO) learns the policy directly but uses V as a *helper* to reduce noise during training. PPO uses both — that's why it has an "actor" (the policy) and a "critic" (the value function).

In your TensorBoard plots, **`explained_variance`** is measuring how well the critic's V predictions match the actual returns. When it's close to 1, the critic is learning. When it dropped on your Pendulum run, the critic was struggling to keep up with the changing policy. That's value functions in action.

---

## Putting it together

Every RL algorithm follows the same skeleton:

1. The agent uses its **policy π** to choose actions based on **states s_t**.
2. The environment returns **rewards R_{t+1}** and **new states s_{t+1}**.
3. Over many episodes, the agent collects **trajectories**.
4. Using those trajectories, the algorithm estimates how good each action was (often via the **value function V** or the **action-value function Q**).
5. The policy is updated to make good actions more likely.

That fifth step is where algorithms diverge. Some update the policy directly (policy gradient methods like PPO). Some derive a policy from a learned Q function (DQN). Some do hybrid things (actor-critic). The next chapter shows the math that ties step 5 together for all of them.

---

## What we did in our project

You've seen all of this concretely:

- **States** — 4 dims (CartPole), 8 dims (LunarLander), 3 dims (Pendulum), 70 dims (Go2)
- **Actions** — discrete in CartPole/LunarLander, continuous Gaussian in Pendulum and Go2
- **Rewards** — survival bonus (CartPole), multi-component shaped (LunarLander), negative-only (Pendulum), and a 20+ term cocktail for the Go2 in walk-these-ways
- **Policies** — Stable-Baselines3's MlpPolicy in Stage 1; the larger pretrained `body_latest.jit` in Stage 2
- **Value functions** — the "critic" network inside SB3's PPO, visible in TensorBoard as explained_variance

The next chapter introduces the math that lets us actually *optimize* policies — Markov Decision Processes, the Bellman equation, and the policy gradient theorem. The friendly math, with every symbol explained.