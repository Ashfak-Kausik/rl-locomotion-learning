# Chapter 5 — PPO In Depth: The Algorithm We Actually Use

## What this chapter is about

You've trained PPO three times in Stage 1, watched its curves, tuned its hyperparameters, and even seen one of its policies collapse on you. Now it's time to know *exactly* what PPO is doing under the hood. By the end of this chapter, every TensorBoard metric you watched will make sense: `clip_fraction`, `approx_kl`, `entropy_loss`, `policy_gradient_loss` — all of them.

PPO (Proximal Policy Optimization, Schulman et al. 2017) is the most widely used RL algorithm today. OpenAI Five used it. ChatGPT's reinforcement learning from human feedback uses it. `walk-these-ways` uses it. Knowing PPO well puts you on solid ground with most of modern RL.

---

## The problem PPO solves

Chapter 4 ended with this: A2C works, but its policy updates can be too aggressive. A single update can change the policy so much that the gradient — which was computed based on the *old* policy — no longer points in a useful direction. Training destabilizes.

The mathematical name for this is the **trust region** problem. You want each update to stay within a "trust region" of the old policy — close enough that the gradient is still informative.

The first attempt to solve this was **TRPO** (Trust Region Policy Optimization). TRPO computes the KL divergence between old and new policies and constrains it directly. It works but is mathematically complex — second-order optimization, conjugate gradients, line search. Painful to implement.

PPO is the "good enough" alternative. Instead of computing KL exactly, it just *clips* the policy ratio. Same goal (keep updates small), much simpler math.

---

## The PPO objective

Here's the equation that defines PPO. We'll unpack it piece by piece.

$$L^{CLIP}(\theta) = \mathbb{E}_t\left[ \min\left( r_t(\theta) \hat{A}_t,\; \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]$$

Where `r_t(θ)` is the **probability ratio**:

$$r_t(\theta) = \frac{\pi_{\theta}(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$$

Translating symbol by symbol:

- **r_t(θ)** is how much more (or less) likely the current policy is to choose action a_t than the old policy was. If the new policy is twice as likely to take that action, r_t = 2.0. If half as likely, r_t = 0.5.
- **Â_t** is the advantage estimate we built up in Chapters 3 and 4.
- **ε** (epsilon) is the **clip range** — a small number, typically 0.2. So `1 − ε = 0.8` and `1 + ε = 1.2`.
- **clip(x, a, b)** is just "if x < a, return a; if x > b, return b; else return x." It bounds the value.

**Reading the equation in plain English:**

For each timestep, compute two candidate update terms:
1. `r_t × Â_t` — the standard policy gradient direction.
2. `clip(r_t, 0.8, 1.2) × Â_t` — the same thing but with the ratio capped between 0.8 and 1.2.

Then take the *minimum* of those two. That minimum is what gets backpropagated.

---

## Why the min and clip work together

This is the clever part. Consider two cases:

**Case 1: Â_t > 0** (the action was good).

You want to make this action more likely. The gradient pushes r_t up — meaning new probability > old probability. But if r_t goes above 1 + ε = 1.2, the clipped term caps it at 1.2 × Â_t while the unclipped term keeps growing. min picks the *smaller* one, so the gradient stops increasing past 1.2. **The policy gets pushed toward this action, but only by a bounded amount.**

**Case 2: Â_t < 0** (the action was bad).

You want to make this action less likely. The gradient pushes r_t down. If r_t goes below 1 − ε = 0.8, the clipped term caps the *penalty* magnitude. But here's the subtle bit: with a negative Â_t, "smaller" means "more negative," so the min picks the *more pessimistic* of the two terms — which is the *unclipped* one. **The policy keeps getting penalized for the bad action, no clipping protection.**

The asymmetry is intentional. Clipping protects the policy from over-updating on good actions (preventing wild lurches toward early-seen-as-good behavior) but doesn't protect bad actions from being driven down. This is the trust region in disguise: updates can shift, but only by `ε` in the favorable direction.

---

## The full PPO loss

The CLIP loss is only part of what PPO actually optimizes. The full objective in your Stable-Baselines3 runs was:

$$L^{PPO} = -L^{CLIP} + c_1 L^{VF} - c_2 H[\pi_{\theta}]$$

Three terms:

1. **L^CLIP** — the clipped policy gradient loss (above). Negated because optimizers minimize, but we want to maximize the objective.
2. **L^VF** — the value function loss. The critic's regression loss for matching observed returns. Weighted by `c_1` (the `value_loss_coef` in your config, default 1.0).
3. **H[π_θ]** — the **entropy** of the policy's action distribution. Subtracted (with `c_2`, the `entropy_coef`) so that *higher entropy is encouraged*. Entropy = how spread out the distribution is. High entropy = exploring; low entropy = committed.

The entropy bonus is RL's anti-laziness mechanism. Without it, a policy might quickly settle on the first decent action it finds and stop exploring. With it, the policy is rewarded for keeping its options open. As training progresses, the policy *naturally* concentrates its probability on good actions, and entropy drops. That's what you saw in Pendulum's `entropy_loss` going from −1 to 0.

---

## The PPO training loop

Here's what happens during `model.learn(total_timesteps=N)`:
for iteration in range(N / n_steps / n_envs):
# 1. ROLLOUT PHASE
for t in range(n_steps):
for each env:
obs = current observation
action, log_prob = policy_old(obs)        # sample action
new_obs, reward, done = env.step(action)
store (obs, action, reward, log_prob, value)

# 2. ADVANTAGE COMPUTATION
Â_t = compute via GAE with current critic V_φ

# 3. POLICY UPDATE
for epoch in range(n_epochs):
    for minibatch in shuffled(rollout_buffer):
        r_t = exp(log π_θ(a) − log π_θ_old(a))
        L_clip = compute clip loss
        L_vf = compute value loss
        L_entropy = compute entropy bonus
        L = L_clip + c1*L_vf − c2*L_entropy
        θ ← θ − α × ∇θ L
        φ ← φ − α × ∇φ L_vf
Three nested loops. The outer iterates over training iterations. The middle collects a rollout. The inner does multiple gradient updates on that same rollout (`n_epochs=10` is typical).

Reusing rollouts multiple times is what makes PPO sample-efficient compared to vanilla A2C — but it's also why clipping is necessary. After 10 epochs of updates, the policy has drifted from the one that collected the rollout, and we need clipping to prevent that drift from breaking things.

---

## Reading TensorBoard metrics with full context

Now every metric you watched in Stage 1 maps to a piece of PPO:

| Metric | What it measures | Healthy range |
|---|---|---|
| `ep_rew_mean` | Average episodic return | Trends up; some wobble OK |
| `mean_value_loss` | How well the critic predicts returns | Drops then plateaus |
| `policy_gradient_loss` | The L^CLIP term's value (not loss to minimize) | Bounded magnitude, doesn't diverge |
| `entropy_loss` | Policy's entropy (negative of) | Drops from high to near 0 over training |
| `approx_kl` | Average KL divergence between old and new policy per update | Stays small, typically < 0.02 |
| `clip_fraction` | Fraction of samples where clipping kicked in | Healthy: 0.1–0.3. Too high (>0.5): updates too aggressive |
| `explained_variance` | How well V_φ predicts actual returns | High = good critic, ~0.9 ideal |

You saw all of these. Your Pendulum run had `clip_fraction` peak at 0.61 — high, which we noted meant the learning rate was slightly too aggressive. That diagnosis came from this table.

---

## What you should remember

- PPO is actor-critic + clipped policy ratio + entropy bonus.
- The clipping prevents the policy from changing too much per update, replacing TRPO's expensive trust-region machinery.
- The min(unclipped, clipped) construction is asymmetric — it protects against over-updating good actions but lets bad actions be penalized freely.
- The full loss combines three terms: policy clip, value regression, entropy bonus.
- Rollouts are reused for multiple gradient epochs per iteration; that's what makes PPO sample-efficient.

PPO is not the most sophisticated RL algorithm — SAC, MuZero, IMPALA all have their use cases. But it hits a remarkably good sweet spot of simplicity, performance, and ease of tuning. That's why it's the workhorse.

---

## What we did in our project

PPO was the only algorithm we used in Stage 1, three times:

- **CartPole** — small policy network (default 64x64), `n_steps=2048`, `n_epochs=10`. Trained at SB3's defaults because the env is forgiving.
- **LunarLander** — `n_steps=1024`, slightly higher `gamma=0.999` because the reward signal is delayed (the +100/-100 only fires at episode end). 8 parallel envs.
- **Pendulum** — `gamma=0.9` because the reward is immediate, `learning_rate=1e-3`, `use_sde=True` for state-dependent exploration in continuous control. This was your first continuous-action policy.

The `walk-these-ways` policy in Stage 2 was also trained with PPO — same algorithm, vastly larger scale (4096 parallel envs in Isaac Gym, 1500 training iterations on GPU). We didn't retrain it; we just used the pretrained weights. But the algorithm that produced those weights is the same one you ran on Pendulum.

Chapter 6 takes everything we know about PPO and starts applying it to a new domain: continuous robot control. That's where the toy-to-robot gap shows up.