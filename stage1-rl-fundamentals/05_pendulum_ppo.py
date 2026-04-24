## Stage 1.3: PPO on Pendulum-v1.
# Goal: Experience a continuous action space environment and see how PPO can handle it.
# This will be the first env where the policy outputs a Gaussian over real-valued actions than discrete probabilities.

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy 

# Create parallel environments
N_ENVS = 8
env = DummyVecEnv([lambda: gym.make("Pendulum-v1") for _ in range(N_ENVS)])
env = VecMonitor(env)   

#Pendulum needs different hyperparameters than LunarLander/CartPole since it's a continuous control task with different reward structure
model = PPO(
    "MlpPolicy",
    env,
    n_steps = 2048,                #more steps per env before updating since the rewards are more sparse and we want more data for stable updates
    batch_size = 64,
    n_epochs = 10,                #more epochs per update to learn more from each batch
    gamma = 0.99,                 #slightly lower gamma since the rewards are more immediate and we want a bit more short-term focus
    gae_lambda = 0.95,
    learning_rate = 1e-3,
    ent_coef = 0.0,               #no entropy bonus since exploration is less of an issue in continuous control with Gaussian policies and it has its own exploration via std
    use_sde=True,                #use State-Dependent Exploration which is often better for continuous control tasks
    verbose = 1,
    tensorboard_log = "./tb_logs/pendulum/"
)

model.learn(total_timesteps = 300_000, progress_bar=True)
model.save("pendulum_ppo")

# Evaluate the trained agent on a fresh single environment
eval_env = gym.make("Pendulum-v1")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
print("Pendulum-v1 is considered solved at around -200 reward, so if you see mean rewards above that, your agent is doing well!")