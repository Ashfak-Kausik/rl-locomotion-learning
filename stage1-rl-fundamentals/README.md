# Stage 1 — RL Fundamentals

This stage covers the foundational projects in Reinforcement Learning, progressively moving from simple discrete control tasks to continuous action spaces. Each project builds on the previous one in terms of complexity, observation space, action space, and training challenges.

---

## Project 1 — CartPole (CartPole-v1)

### What it is
A pole is balanced on a cart. The agent must move the cart left or right to keep the pole from falling over.

### Specs
- **Observations:** 4 (cart position, cart velocity, pole angle, pole angular velocity)
- **Actions:** 2 discrete (move left, move right)
- **Reward:** +1 for every step the pole stays upright
- **Solved at:** mean reward ≥ 475 (max is 500)

### What it covered
- First introduction to the PPO (Proximal Policy Optimization) algorithm.
- Understanding what a policy is — a function that maps observations to actions.
- Understanding the training loop: observe → act → get reward → update policy.
- Reading TensorBoard metrics for the first time (`ep_rew_mean`, `explained_variance`, `entropy_loss`, etc.).
- Understanding evaluation vs training — freezing the agent's brain and testing performance.
- The difference between the Gym API (`obs, info = env.reset()`) and the SB3 VecEnv API.
- Saving and loading trained models.

### What happened
Trained for 100,000 timesteps. The agent went from a mean reward of ~21 (barely surviving) to a consistent mean reward of ~500 (perfectly balancing the pole indefinitely). The `ep_rew_mean` curve showed a clean S-curve of learning — flat early, steep middle, plateau at the top. A perfect first RL result.

---

## Project 2 — LunarLander (LunarLander-v3)

### What it is
A spacecraft must land safely on a landing pad between two flags. The agent controls three engines (left, right, main) to guide the lander down without crashing.

### Specs
- **Observations:** 8 (x/y position, x/y velocity, angle, angular velocity, leg contact left/right)
- **Actions:** 4 discrete (do nothing, fire left engine, fire main engine, fire right engine)
- **Reward:** +100–140 for landing, bonus for fuel efficiency, large penalty for crashing
- **Solved at:** mean reward ≥ 200

### What it covered
- A significantly harder environment than CartPole — more observations, more actions, more complex reward structure.
- Seeing how training instability looks in practice — the agent peaked around 850k steps then experienced **policy collapse**, where continued training actually degraded performance.
- Understanding why more training is not always better, and why saving the best checkpoint during training (using `EvalCallback`) matters.
- Comparing multiple PPO runs (PPO_1 through PPO_4) on TensorBoard and seeing how different training lengths affect all metrics.
- Understanding `explained_variance` as a measure of how accurately the value function predicts future rewards, and watching it drop when the policy destabilizes.

### What happened
Trained for 300k, 500k, and 1M timesteps across multiple runs. The 1M run peaked at around 860k steps with `explained_variance` near 0.96 and consistent landings, then collapsed — `ep_len_mean` dropped, rewards became inconsistent, and `explained_variance` fell to ~0.6. Evaluation showed mean rewards of 220–274, well above the solved threshold of 200, confirming the agent genuinely learned to land. The sweet spot was estimated at around 700k–800k steps.

---

## Project 3 — Pendulum (Pendulum-v1)

### What it is
An underpowered pendulum must be swung up from a hanging position and held upright. Unlike CartPole, the agent cannot simply push left or right — it must apply a continuous torque value anywhere between -2 and +2 Nm. There is no termination condition; every episode runs for exactly 200 steps regardless of performance.

### Why is it fundamentally different?

| | CartPole | LunarLander | Pendulum |
|---|---|---|---|
| Action space | Discrete (2) | Discrete (4) | **Continuous** |
| Reward direction | Positive (↑ better) | Positive (↑ better) | **Negative (→ 0 better)** |
| Episode end | Pole falls | Lands or crashes | **Always 200 steps** |
| Solved at | ≥ 475 | ≥ 200 | **≥ -200** |
| Difficulty | Easy | Medium | Hard |

This is the first environment with a **continuous action space** — the agent doesn't choose from a list of actions, it outputs an exact real-valued torque. This introduces an entirely new challenge: the policy must now learn a probability distribution over a continuous range, not just pick from a small set of options.

### New concepts introduced

**Continuous action space**
Instead of "go left or go right", the agent outputs a precise torque between -2 and +2. This requires the policy to output a mean and a standard deviation, forming a Gaussian distribution it samples actions from. Early in training this distribution is wide (high std = lots of randomness). As the agent learns, the std shrinks as it becomes more confident in its torque choices.

**Negative reward system**
Pendulum rewards are always negative, calculated as a penalty based on how far the pendulum is from upright, how fast it's moving, and how much torque was applied. A perfectly balanced pendulum with no torque applied scores 0 — which is theoretically impossible to achieve in practice. A random agent scores around -1400 to -1600. The goal is to get as close to -200 as possible.

**`train/std` metric**
This metric only appears in continuous action space environments. It measures how wide the agent's action distribution is — how much randomness it adds to its torque decisions. In this run it started at ~0.95 (very exploratory) and dropped to ~0.39 by 100k steps, showing the agent was becoming more decisive and precise about what torque to apply.

### Training approach — iterative experimentation

Rather than training once, we ran Pendulum at multiple timestep budgets deliberately — to observe how the metrics evolved and to understand at what point the agent actually converges. This is a key skill in RL: knowing when to stop, when to keep going, and what the curves are telling you.

| Run | Timesteps | Mean Reward | Notes |
|---|---|---|---|
| PPO_1 | 100,000 | -778.98 | Still learning, not converged |
| PPO_2 | 300,000 | ≥ -200 ✅ | Solved — agent consistently balancing |
| PPO_3 | 500,000 | Solved+ | Diminishing returns, policy stabilized |
| PPO_4 | 1,000,000 | Solved+ | Some interesting outputs in the tensorboard visuals after 800K. For instance, in the ep_lean_mean: The curve rises beautifully to ~900-1000 steps, then drops back down around 850k steps. Likewise there were some other interesting findings in other metrics too. Almost all centered around the term "policy collapse or overtraining instability" which is- PPO keeps updating the policy even when it's already good. Small updates can accidentally "nudge" the policy away from what was working. The agent unlearns a subtle part of its strategy and episodes get shorter again. |

**The key finding: 300k was the sweet spot.** Beyond that, the agent had already converged and additional training added no meaningful improvement — unlike LunarLander where overtraining caused actual degradation, Pendulum simply plateaued.

### Training configuration (final solved run)

```python
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.001,
    n_steps=2048,
    verbose=1,
    tensorboard_log="./tb_logs/pendulum/"
)
model.learn(total_timesteps=300_000)
```

### Training results and interpretation (100k observation run)

The 100k run was used to study and interpret the metrics in detail, since the curves were still actively changing and easier to read. Here is what each metric showed:

**`ep_rew_mean`: -1400 → -1050**
Reward improved significantly, moving in the right direction (toward 0). The curve was still rising at the end of training — the agent had not yet converged, confirming more timesteps were needed.

**`ep_len_mean`: flat at 200**
Completely uninformative for Pendulum. Every episode is exactly 200 steps by design — there is no termination condition. This graph should be ignored for this environment entirely.

**`explained_variance`: ~0.20 → 0.66**
Rose steadily but only reached 0.66 at 100k steps — significantly lower than LunarLander which reached 0.96. By 300k it reached 0.9+, confirming the value function needed more time. A still-rising explained variance curve is a reliable signal that training isn't done yet.

**`train/std`: 0.95 → 0.39**
Clean downward curve throughout training — the agent was consistently becoming more confident and precise in its torque decisions. This metric only appears in continuous action space environments. Starting high means the agent was exploring broadly; shrinking std means it was committing to learned strategies.

**`train/entropy_loss`: -2.55 → -1.97**
Rose steadily (became less negative) — the agent moved from random exploration toward more decisive, committed actions. The values here are far more negative than CartPole or LunarLander because continuous action spaces have inherently higher entropy — there are infinitely many possible torque values versus 2 or 4 discrete choices.

**`train/value_loss`: ~2500 → ~297**
Dropped sharply and was still falling at 100k steps. By 300k it fully plateaued, confirming convergence. A value loss that is still actively dropping at the end of training is one of the clearest signs that more timesteps are needed.

**`train/clip_fraction`: ~0.45–0.50 ⚠️**
The most concerning early metric. The healthy range is 0.1–0.3. At 0.45–0.50, nearly half of all policy updates were being clipped by PPO's safety mechanism — the policy was trying to change too aggressively per update. This is a symptom of the learning rate being slightly high for this environment. Despite this, the agent still converged at 300k, showing PPO is robust enough to handle some instability.

**`train/approx_kl`: ~0.08–0.09 ⚠️**
Healthy range is below 0.02. At 4x the healthy threshold, this confirmed the same issue as clip_fraction — updates were larger than ideal. Watching these two metrics together is the fastest way to diagnose learning rate problems.

**`train/policy_gradient_loss`: -0.04 → -0.079**
Becoming more negative over time, indicating the policy was still making significant corrections at 100k. By 300k this stabilized, consistent with full convergence.

**Evaluation result at 100k (before solving)**
```
Mean reward: -778.98 +/- 57.89
```
Roughly halfway between a random agent (-1400 to -1600) and solved (-200). The std of 57.89 shows reasonable consistency — the agent had a strategy, just not a good enough one yet.

**Evaluation result at 300k (solved)**
```
Mean reward: ≥ -200
```
Agent successfully swings the pendulum up and holds it upright consistently across episodes.

### Key lessons from Pendulum

- **Continuous action spaces are fundamentally harder than discrete ones.** The agent must learn not just what direction to act but exactly how much force to apply at every moment.
- **Negative reward environments require a mental shift.** Progress means the number getting less negative, not larger. A result of -778 is genuinely good progress from -1400.
- **Iterative experimentation is how real RL works.** Running at 100k, 300k, 500k, and 1M and comparing the results taught more than any single run would have. You learn to read when a model is done versus when it needs more time.
- **300k was the sweet spot — more timesteps added nothing.** Unlike LunarLander where overtraining caused collapse, Pendulum simply plateaued. Knowing the difference matters.
- **`clip_fraction` and `approx_kl` are early warning signs.** High values tell you the learning rate is too aggressive before you even need to look at reward curves.
- **A still-changing curve = training isn't done.** When `value_loss`, `std`, and `ep_rew_mean` are all still actively moving at the end of training, the agent hasn't finished learning — give it more timesteps.

---

## Overall Stage 1 Summary

| Project | Algorithm | Action Space | Timesteps to Solve | Final Mean Reward | Solved? |
|---|---|---|---|---|---|
| CartPole-v1 | PPO | Discrete (2) | 100,000 | ~500 | ✅ Yes |
| LunarLander-v3 | PPO | Discrete (4) | ~700k–800k | ~220–274 | ✅ Yes |
| Pendulum-v1 | PPO | Continuous | 300,000 | ≥ -200 | ✅ Yes |

All three environments were solved. Stage 1 established the complete foundation for RL experimentation: setting up environments, training with PPO, reading and interpreting TensorBoard metrics, saving and loading models, watching trained agents, and understanding the difference between discrete and continuous control problems. The progression from CartPole → LunarLander → Pendulum deliberately increased complexity at each step, exposing a new layer of RL concepts with each environment. The Pendulum project in particular introduced the habit of iterative experimentation — running multiple training lengths and comparing results — which is how real RL research and development works.
