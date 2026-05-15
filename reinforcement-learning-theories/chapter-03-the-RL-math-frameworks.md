# Chapter 3 — The Math Behind It: MDPs, Bellman, and Advantages

## What this chapter is about

This is the math chapter. Don't skip it — but also don't memorize it. The goal here is to give you enough mathematical fluency that when you see equations in RL papers, you can read them like sentences. Every symbol gets named, every formula gets a worked example, and every concept gets a translation back to what you already know.

By the end you'll know what an MDP is, why the Bellman equation is the most important equation in RL, and what an "advantage" is — the quantity that PPO uses to decide how much to nudge the policy.

---

## Markov Decision Processes (MDPs)

Every RL problem is mathematically described as an **MDP** — a Markov Decision Process. It's a 5-tuple:

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$$

Translating that symbol by symbol:

- **S** is the set of all possible states. For CartPole, S is all valid (position, velocity, angle, angular velocity) tuples. For the Go2, S is the 70-dimensional observation space.
- **A** is the set of all possible actions. {left, right} for CartPole. ℝ¹² for the Go2.
- **P** is the **transition function**: `P(s' | s, a)` = the probability that taking action `a` in state `s` leads to state `s'`. In a deterministic environment, P is just a function. In a stochastic one (most real systems), it's a probability distribution.
- **R** is the **reward function**: `R(s, a, s')` = the reward received when transitioning from `s` to `s'` via `a`.
- **γ** is the **discount factor** we met in Chapter 1 and 2.

The "Markov" in MDP refers back to the Markov property: the next state depends *only* on the current state and action, not on the history. This is why we said in Chapter 2 that observations need to be Markovian — the entire mathematical framework assumes it.

**Why this matters.** When you read an RL paper that says "we model the problem as an MDP with state space S and action space A," they're telling you exactly what S, A, P, R, and γ are. That's the contract. Everything that follows is built on top.

---

## The Bellman equation

If there's one equation to know in RL, it's this one. Recall the value function from Chapter 2:

$$V^{\pi}(s) = \mathbb{E}_{\pi}\left[ G_t \mid s_t = s \right]$$

The **Bellman equation** is a recursive way of writing the same thing:

$$V^{\pi}(s) = \mathbb{E}_{\pi}\left[ R_{t+1} + \gamma V^{\pi}(s_{t+1}) \mid s_t = s \right]$$

In plain English: *the value of a state equals the immediate reward plus the discounted value of wherever you end up next.*

This is the recursive insight that makes RL tractable. Instead of computing the return as a long discounted sum (which requires playing the whole future forward), you can compute it locally: just need to know the immediate reward and the value of the next state.

**A worked example.** Imagine a tiny 3-state world: A → B → C (terminal). Rewards are R(A→B) = 1, R(B→C) = 10. Discount γ = 0.9. Deterministic transitions.

- V(C) = 0 (terminal)
- V(B) = 10 + 0.9 × V(C) = 10 + 0 = 10
- V(A) = 1 + 0.9 × V(B) = 1 + 9 = 10

Notice we computed values "backward" from the terminal state. That's called *bootstrapping*, and it's how most RL algorithms estimate V — each value estimate uses other value estimates.

When you saw `explained_variance` in your Pendulum TensorBoard plots, the critic was estimating V(s) for every state visited, and explained_variance measured how well those estimates matched the actual G_t observed in the rollouts. If the critic uses bootstrapping (which PPO does, via *Generalized Advantage Estimation*), the estimates are self-consistent: V(s) = R + γV(s').

---

## The action-value Bellman equation

Same idea, but for Q (the action-value function from Chapter 2):

$$Q^{\pi}(s, a) = \mathbb{E}\left[ R_{t+1} + \gamma \mathbb{E}_{a' \sim \pi}\left[ Q^{\pi}(s_{t+1}, a')\right] \right]$$

That nested expectation looks scary but says: *the value of taking action `a` in state `s` equals the immediate reward, plus γ times the average Q value at the next state, averaged over what action your policy would take there.*

If you're learning Q directly (as in Q-learning or DQN), this is the equation you're trying to satisfy. You collect transitions, you measure `R + γ max_a' Q(s', a')`, and you update Q to be closer to that target. Bellman is the entire algorithm in one line.

---

## The advantage function

Now the concept that makes PPO tick: the **advantage**.

$$A^{\pi}(s, a) = Q^{\pi}(s, a) - V^{\pi}(s)$$

Read it: *the advantage of action `a` in state `s` is how much better that action is than the policy's average behavior in `s`.*

A worked example. Suppose at state `s`, your policy chooses three actions with these Q values: Q(s, a₁) = 8, Q(s, a₂) = 10, Q(s, a₃) = 12. And suppose the policy picks each with equal probability, so V(s) = 10.

- A(s, a₁) = 8 − 10 = −2 (worse than average)
- A(s, a₂) = 10 − 10 = 0 (average)
- A(s, a₃) = 12 − 10 = +2 (better than average)

**Why advantage matters more than reward.** Imagine two states. State `s_easy` always gives high reward regardless of action. State `s_hard` always gives low reward regardless of action. If you naively trained your policy on "high reward = good action," you'd learn that *any* action in `s_easy` is good — but that's not learning, that's just noting which states are easy.

Advantage strips this away. It asks: *among the actions available here, which was better than average?* That's the signal you actually want to push your policy toward.

PPO and most modern policy-gradient algorithms train on advantages, not raw rewards. When you saw `mean_surrogate_loss` and `policy_gradient_loss` in TensorBoard, advantages were what those losses were computed from.

---

## Generalized Advantage Estimation (GAE) — brief mention

Computing the "true" advantage requires perfect knowledge of Q and V, which you don't have during training. So algorithms estimate it from sampled trajectories.

The simplest estimate is the **TD residual**: δ_t = R_{t+1} + γV(s_{t+1}) − V(s_t). This is "how much better did things turn out than V predicted?"

**GAE** averages TD residuals over multiple future steps with another discount factor λ (lambda):

$$\hat{A}_t^{GAE} = \sum_{k=0}^{\infty} (\gamma \lambda)^k \delta_{t+k}$$

When λ = 0, you get pure TD (low variance, high bias). When λ = 1, you get the Monte Carlo return (high variance, low bias). The `gae_lambda=0.95` in your PPO config was tuning this tradeoff — slightly biased estimates with manageable variance.

You don't need to memorize the GAE formula. You just need to know: PPO uses GAE under the hood to compute advantages, and `gae_lambda` is the dial.

---

## Why this math matters

Every concept in this chapter shows up in PPO's update rule, which we'll see in Chapter 5. Specifically:

- **MDPs** define what problem you're solving.
- **Value functions** are what the critic network estimates.
- **The Bellman equation** is how the critic is trained (its target is `R + γV(s')`).
- **Advantages** are what the policy update is multiplied by — bigger positive advantage = bigger nudge to make that action more likely.
- **GAE** is the specific way PPO estimates those advantages.

If you understood this chapter, the next two will feel like applications of these ideas rather than new mysteries.

---

## What we did in our project

The math from this chapter was working invisibly in every PPO run:

- **CartPole/LunarLander/Pendulum** are all MDPs. We described their S, A, R, γ implicitly via Gymnasium's `env.observation_space`, `env.action_space`, and the env's internal reward function. γ was the `gamma` parameter you set (0.99, 0.999, 0.9 for the three envs).
- **The critic** inside SB3's PPO was estimating V(s) for every state encountered. The "value loss" TensorBoard metric was tracking how well it satisfied Bellman.
- **Advantages** were computed via GAE with λ=0.95. The PPO policy update was: `(new action probability / old action probability) × advantage`, clipped to prevent runaway updates. (We unpack the clipping in Chapter 5.)
- **The `approx_kl` metric** measured how much each PPO update changed the policy, which is closely tied to how aggressively the advantages pushed the policy.

The next chapter (4) puts the math to work in the simplest possible algorithm — REINFORCE — and shows how that primitive grows into modern actor-critic methods like A2C, the direct ancestor of PPO.