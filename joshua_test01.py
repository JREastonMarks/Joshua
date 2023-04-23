import gymnasium as gym
from compute.brain import Brain
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()

brain = Brain(default_action=env.action_space.sample(), cosine_cutoff=0.87)


for _ in range(100):
    # action = env.action_space.sample()  # agent policy that uses the observation and info
    
    action = int(brain.step(observation)[0][0])
    observation, reward, terminated, truncated, info = env.step(action)

    brain.update(observation, reward)

    if terminated or truncated:
        observation, info = env.reset()

env.close()