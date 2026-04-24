import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("Pendulum-v1", render_mode="human")
model = PPO.load("pendulum_ppo")

obs, _ = env.reset()
total_reward = 0
episodes = 0
while episodes < 5:                                      # Watch 5 episodes
    action, _ = model.predict(obs, deterministic=True)   # Get action from the trained model
    obs, reward, terminated, truncated, info = env.step(action)  # Take a step in the environment
    total_reward += reward
    if terminated or truncated:                          # Check if episode is done
        print(f"Episode {episodes + 1} finished with total reward: {total_reward:.2f}")
        obs, _ = env.reset()                             # Reset the environment for the next episode
        total_reward = 0
        episodes += 1
env.close()