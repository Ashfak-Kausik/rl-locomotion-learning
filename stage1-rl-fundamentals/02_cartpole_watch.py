## Let's load the CartPole policy and watch it in action!
from turtle import done

import gymnasium as gym
from stable_baselines3 import PPO 

env = gym.make("CartPole-v1", render_mode="human")  # Create environment with rendering enabled
model = PPO.load("ppo_cartpole")  # Load the trained PPO model  

obs, info = env.reset()  # Reset the environment to start a new episode
for _ in range(10000):
    action, _ = model.predict(obs, deterministic=True) # Get the action from the model's policy based on the current observation
    obs, reward, terminated, truncated, _ = env.step(action) # Take an action in the environment and observe the results
    if terminated or truncated:
        obs, info = env.reset()  # Reset the environment if the episode ends (pole falls or cart goes out of bounds)
env.close() # Close the environment when done 