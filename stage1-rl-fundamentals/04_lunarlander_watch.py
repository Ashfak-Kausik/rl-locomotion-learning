import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("LunarLander-v3", render_mode="human")
model = PPO.load("lunarlander_ppo")

obs, _ = env.reset()
total_reward = 0
episodes = 0
while episodes < 5:                                              # Watch 5 episodes of the trained agent
    action, _ = model.predict(obs, deterministic=True)           # Get action from the trained model
    obs, reward, terminated, truncated, info = env.step(action)  # Take action in the environment
    total_reward += reward
    if terminated or truncated:                                  # Check if the episode is done
        print(f"Episode {episodes + 1} finished with reward: {total_reward:.2f}")
        total_reward = 0
        episodes += 1
        obs, _ = env.reset()                                     # Reset the environment for the next episode
env.close()