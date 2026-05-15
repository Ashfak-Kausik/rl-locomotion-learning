# Chapter 4 — Policy Gradient Methods: From REINFORCE to A2C

## What this chapter is about

Chapter 3 gave us the math objects: value functions, Bellman equations, advantages. Now we use them. This chapter walks through three algorithms in order of increasing sophistication, each fixing a problem in the one before it:

1. **REINFORCE** — the simplest policy gradient method. Works but is wildly noisy.
2. **REINFORCE with baseline** — subtracts a baseline to reduce noise. Conceptual ancestor of using value functions in training.
3. **Actor-Critic / A2C** — uses a learned value function as the baseline. Where modern RL really begins.

PPO is the direct descendant of A2C and gets its own chapter next. So treat this as the "how did we get to PPO" tour.

---

## The policy gradient theorem (in one paragraph)

Here's the deep math result that powers everything. If your policy π is parameterized by θ (theta) — the neural network weights — and your goal is to maximize expected return J(θ), then the gradient of J with respect to θ is:

$$\nabla_{\theta} J(\theta) = \mathbb{E}_{\tau \sim \pi}\left[ \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t | s_t) \cdot G_t \right]$$

Don't panic. The whole equation says: *to improve the policy, sample some trajectories, then for each action taken, nudge the policy weights in the direction that makes that action more likely if the return was high, and less likely if the return was low.* That's it.

The `∇_θ log π_θ(a_t | s_t)` term is just the math for "how does the policy's probability of choosing action a_t change as we tweak the weights θ?" Modern deep learning frameworks (PyTorch, JAX) compute this for you automatically. You don't have to.

**The key insight:** good actions get reinforced, bad actions get suppressed, and the strength of the update is proportional to the return. That's policy gradient, in one sentence.

---

## REINFORCE — the simplest version

REINFORCE (introduced by Williams in 1992) applies the policy gradient theorem literally:

1. Run the current policy for an episode, recording (s_t, a_t, R_{t+1}).
2. Compute the return G_t for each timestep.
3. Update θ: `θ ← θ + α × G_t × ∇_θ log π_θ(a_t | s_t)` for every timestep.
4. Repeat.

That `α` is the learning rate. The whole algorithm is the policy gradient theorem applied verbatim.

**Why it works.** If action a_t led to a high G_t, the gradient nudges θ to make a_t more likely in state s_t in the future. If a_t led to low G_t, the gradient nudges θ to make a_t less likely.

**Why it's bad.** Three problems:

1. **High variance.** G_t can swing wildly between episodes. Even with a fixed policy, the same starting state can lead to vastly different returns. The gradient is noisy, training is slow.
2. **No credit assignment.** All actions in a trajectory get scaled by the same G_t, so a great action followed by a terrible one might both get suppressed if the overall return was poor.
3. **Sample-inefficient.** You have to wait until the end of an episode to compute G_t, then throw away the episode and collect another. Expensive.

REINFORCE is rarely used in practice today. But it's the conceptual foundation everything else builds on.

---

## REINFORCE with baseline

Here's a clever fix for the variance problem. Notice that you can subtract any function `b(s_t)` from the return without changing the gradient's expectation (this is a mathematical fact about gradients and expectations):

$$\nabla_{\theta} J(\theta) = \mathbb{E}\left[ \sum_t \nabla_{\theta} \log \pi_{\theta}(a_t | s_t) \cdot (G_t - b(s_t)) \right]$$

The function `b(s_t)` is the **baseline**. As long as it doesn't depend on the action, the math still works — but the variance drops dramatically.

What's a good baseline? The value function V(s)! It represents the "average return I expect from this state." If G_t > V(s_t), this action did better than average → push it up. If G_t < V(s_t), it did worse than average → push it down. The quantity `G_t − V(s_t)` is exactly the *advantage estimate* from Chapter 3.

This is the moment value functions stop being just an analysis tool and become an active component of training.

---

## Actor-Critic / A2C

Now we generalize the previous idea into a proper algorithm: maintain two networks side by side.

- The **actor** is the policy π_θ — outputs actions. This is what you eventually deploy.
- The **critic** is the value function V_φ — outputs estimated values. This trains alongside the actor, only to help it learn.

(θ and φ are just letters — Greek phi — for the two sets of weights. The critic has its own parameters, independent of the actor.)

The algorithm:

1. Run the actor for some number of steps (not necessarily a full episode).
2. Use the critic to compute advantage estimates Â_t.
3. Update the actor in the direction of the policy gradient, weighted by Â_t.
4. Update the critic to better predict the actual returns (this is a regression problem — make V_φ(s_t) match the observed return).
5. Repeat.

**A2C** (Advantage Actor-Critic) is the synchronous version of this, where multiple environments are run in parallel and their experiences are averaged together for each update. The "synchronous" is in contrast to A3C, which used asynchronous updates and is mostly historical now.

**Why this is a real improvement:**

- The critic learns the easy thing (predicting returns) so the actor can focus on the hard thing (choosing actions).
- The critic provides a baseline at every state, not just an average — dramatically reduces variance.
- You can update before episodes end (using bootstrapped value estimates), which is more sample-efficient.

Most modern RL algorithms (PPO, SAC, DDPG, TD3) are variants of actor-critic. The actor-critic architecture is the dominant paradigm.

---

## A tour stop before PPO

A2C works but has its own problem: **updates can be too aggressive**. If a particular Â_t is large, the policy gradient pushes π_θ hard in one direction. Sometimes too hard — the policy changes so much in one step that the *new* policy is collecting completely different trajectories than the *old* one, and the gradient becomes misleading. Training destabilizes. You see your `ep_rew_mean` curve climb beautifully, then crash.

PPO's entire contribution is fixing this. It adds a clipping mechanism that prevents the policy from changing too much in one update. We'll see exactly how in Chapter 5.

But every other piece of PPO is already here in this chapter:

- The policy gradient ∇log π × Â
- Advantage estimates (from a critic network)
- Actor-critic architecture
- Synchronous parallel envs (your `DummyVecEnv(8)` in Stage 1)

PPO is genuinely just "A2C with a clipping trick + a few tuning knobs." That's why it's the default — it captures 80% of what works in modern RL with 20% of the complexity.

---

## What we did in our project

Every PPO training run in Stage 1 was secretly running this entire chapter:

- The **actor** was the MlpPolicy that mapped observations to action probabilities (CartPole/LunarLander) or to Gaussian parameters (Pendulum).
- The **critic** was a second MLP with the same input but output dim 1 — that's V_φ(s).
- **Advantages** were estimated via GAE (Chapter 3), giving `Â_t` for every timestep.
- **`n_steps=2048`** meant: each environment runs the actor for 2048 timesteps, then the actor and critic are updated using the collected batch.
- **`mean_value_loss`** in TensorBoard was the critic's regression loss — how badly it was predicting actual returns.
- **`mean_surrogate_loss`** was the actor's policy gradient loss — the thing the actor was minimizing (well, technically maximizing).

The `clip_fraction` and `approx_kl` metrics you watched are the PPO clipping mechanism kicking in — but for that, we need Chapter 5.