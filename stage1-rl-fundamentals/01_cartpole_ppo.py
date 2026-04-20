## First Reinforcement Learning Project: CartPole with PPO (Cartpole-v1)

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

## Let's create the Cartpole environment
# A cartpole balances a pole on a cart and the goal is to keep the pole balanced by moving the cart left or right.
# Observations: 4 (cart position, cart velocity, pole angle, pole velocity at tip/angular velocity)
# Actions: 2 (move cart left or right)
# Rewards: +1 for every step the pole is balanced, episode ends when pole falls or cart goes out of bounds.
env = gym.make("CartPole-v1")

## Let's create the PPO (Proximal Policy Optimization) agent
# "MlpPolicy" = Multi-layer Perceptron policy (fully connected neural network)
# "verbose=1" = print training progress
# tensorboard_log creates a directory for TensorBoard logs to visualize training metrics
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./tb_logs/cartpole/"
)

## Let's train the agent for 10,000 timesteps
model.learn(total_timesteps=100000)

## Save the trained model to a file named "ppo_cartpole"
model.save("ppo_cartpole")

## Let's evaluate the trained agent
# "eval_env" = environment used for evaluation (can be the same as training environment)
# "n_eval_episodes" = number of episodes to evaluate the agent on
# "return_episode_rewards=True" = return rewards for each episode instead of mean reward
mean_reward, std_reward = evaluate_policy(
    model,
    gym.make("CartPole-v1"),
    n_eval_episodes=10,
    return_episode_rewards=False
)
print(f"Mean reward: {mean_reward} +/- {std_reward}")
print("CartPole-v1 is 'solved' at means reward >= 475 (max is 500).")

