# Chapter 1 — Introduction to Reinforcement Learning

## What this chapter is about

Welcome. If you're brand new to RL, this is your starting line. By the end of this chapter, you'll know what RL actually *is*, why it exists as a separate field from regular machine learning, where it's useful, and the rough shape of every RL system you'll ever build. No prerequisites beyond basic programming and a willingness to think.

We'll keep things conversational. Where math shows up — and it will, gently — every symbol gets explained. Nothing assumed.

---

## The core idea, in one sentence

**Reinforcement Learning is how an agent learns to make good decisions by trying things, seeing what happens, and gradually doing more of what works.**

That's it. Everything else in this field is engineering details around that idea.

But "what works" is doing a lot of heavy lifting in that sentence. Let's unpack it.

---

## Why RL exists as its own thing

You've probably heard of *supervised learning* — show a model thousands of cat photos labeled "cat," and it learns to recognize cats. That works beautifully when you have labeled data.

But what if you don't? What if your problem is something like:

- Teach a robot to walk
- Play chess at a grandmaster level
- Decide when to buy or sell a stock
- Drive a car

You can't make a labeled dataset for "the correct action when the chessboard looks like this," because there is no single correct answer — it depends on what happens next, what your opponent does, what your long-term plan is. The "label" only emerges after a sequence of decisions plays out.

This is the **sequential decision problem**. And it's what RL was built for.

In RL, instead of telling the model the right answer, you give it a *reward signal* — a number that tells it whether things are going well or badly. The model has to figure out, through trial and error, which actions lead to high rewards over time.

It's closer to how animals learn than how you'd train a classifier. A puppy doesn't learn "sit" from labeled examples. It tries random things, gets a treat when it accidentally sits, and over time the connection forms.

---

## The five things every RL system has

Every RL setup, no matter how complex, has these five components. Memorize these names:

1. **Agent** — the thing that's learning and making decisions. The puppy. The chess engine. The robot.
2. **Environment** — the world the agent interacts with. The owner with treats. The chessboard. The physics simulator.
3. **State** (sometimes called *observation*) — what the agent perceives about the world right now. The puppy sees the owner's hand gesture. The chess engine sees the board. The robot sees its joint angles and gravity vector.
4. **Action** — what the agent does. The puppy sits or stays standing. The engine moves a piece. The robot sends torques to its motors.
5. **Reward** — a number the environment gives back after each action. Positive = good, negative = bad. The treat. The piece advantage. The robot's forward velocity.

The loop is dead simple:
state → agent chooses action → environment changes → new state + reward → repeat

This loop runs over and over. The agent's job is to learn a **policy** — a strategy that maps states to actions — that maximizes the total reward collected over time.

That's it. That's RL.

---

## A tiny example

Imagine a robot in a 4×4 grid. There's a goal in the top-right corner. The agent can move up, down, left, or right.

- **State:** the robot's current (x, y) position
- **Action:** {up, down, left, right}
- **Reward:** +10 if it reaches the goal, −1 for every other step, −100 if it falls off the grid

The robot starts knowing nothing. It tries random moves. Most lead to −1, some lead to −100, a few lucky sequences reach the goal and get +10. Over many tries, it learns: "from (3, 3), going up gives me a chance at +10. From (0, 0), going up-then-up tends to do better than down-then-down."

After enough trials, the robot develops a policy: a learned mapping from "current position" to "best direction to go." This is RL in miniature.

Now scale this up. Replace "4×4 grid" with "12-joint quadruped in continuous space." Replace "(x, y)" with "70-dimensional observation vector." Replace "{up, down, left, right}" with "12 continuous joint commands." Replace "reach the goal" with "track a desired velocity while staying upright and stepping rhythmically." Same loop. Same idea. Much harder.

---

## A first taste of the math

The agent's goal is to maximize the **return** — the total reward collected over time. We write it like this:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots$$

Don't panic. Here's what each symbol means:

- `G_t` is the return starting from time step `t`
- `R_{t+1}` is the reward received at the next step
- `γ` (gamma) is the **discount factor**, a number between 0 and 1 (typically 0.99)
- The exponents on γ make future rewards count slightly less than immediate ones

**Why discount?** Because if a robot only cares about the very long term, it might do nothing useful right now. The discount factor says "near-future rewards matter more than far-future ones." It's the math equivalent of "a bird in the hand is worth two in the bush."

You'll see this exact formula again in every RL algorithm. It's the foundation.

---

## Where RL fits in the bigger picture

A quick map:

| Type of ML | What it needs | Example |
|---|---|---|
| Supervised learning | Labeled examples | Image classification |
| Unsupervised learning | Unlabeled data, find structure | Clustering customers |
| Reinforcement learning | A reward signal and an environment | Robot walking, game playing |

RL is genuinely different. You don't need a dataset — you need a *world* the agent can interact with, and a way to score what happens. That's why most RL today happens in simulation: it's the cheapest, fastest, and safest "world" to provide.

---

## What we did in our project

In this project, RL shows up first in **Stage 1**, where we trained PPO (a specific RL algorithm) on three toy environments:

- **CartPole** — balance a pole on a cart. State = 4 numbers. Action = push left or push right. Reward = +1 per step the pole stays up.
- **LunarLander** — land a spacecraft between two flags. State = 8 numbers. Action = 4 choices for which thruster to fire. Reward = a mix of landing softly, fuel cost, position.
- **Pendulum** — swing a pendulum upright. State = 3 numbers. Action = a continuous torque value. Reward = always negative, closer to zero = better.

These three problems are the "hello world" of RL. They each teach a different lesson:

- CartPole: how the basic loop works
- LunarLander: how shaped rewards behave
- Pendulum: how to handle continuous actions (which is the prerequisite for robot control, since joints are continuous)

Once those clicked, we moved to **Stage 2**, where the agent became a quadruped robot, the environment became MuJoCo, the state grew to 70 dimensions, and the action became 12 continuous joint commands. The same five-component framework. Much bigger numbers.

The next chapter goes deeper on each of those five components — what they look like in practice, what makes them tricky to design, and the vocabulary you'll need to understand modern RL papers.
