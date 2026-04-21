## Second reinforcement learning project: LunarLander PPO
#Goal: Experience a harder environment and implement a more complex algorithm (shaped rewards and dense failure modes)

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy

# Creating paralell environments
N_ENVS = 8
env = DummyVecEnv([lambda: gym.make("LunarLander-v3") for _ in range(N_ENVS)])
env = VecMonitor(env)   # tracks episode rewards and lengths for logging purposes

# Create the PPO agent (PPO with slightly tuned hyperparameters)
# These are the SB3 RL Zoo defaults for this env (hand-tuned by community)
model = PPO(
    "MlpPolicy",
    env,
    n_steps = 1024,                #steps per env before updating the policy (so total steps = n_steps * n_envs = 8192)
    batch_size = 64,                   
    n_epochs = 4,
    gamma = 0.999,                 #high gamma means future rewards are more important (long-term focus)
    gae_lambda = 0.98,
    ent_coef = 0.01,               #small entropy bonus encourages some exploration but not too much
    learning_rate = 3e-4,
    verbose = 1,
    tensorboard_log = "./tb_logs/lunarlander/"
)

# Train the Lunarlander PPO agent for 300K-1M timesteps to see what happens 
# Mostly 500K to 1M timesteps is enough to train a LunarLander PPO agent to a good level of performance.
model.learn(total_timesteps = 500_000, progress_bar=True)

model.save("lunarlander_ppo")

# Evaluate the trained agent on a fresh single environment
eval_env = gym.make("LunarLander-v3")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")