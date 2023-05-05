import gymnasium as gym
from compute.brain import Brain
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()

brain = Brain(episodic_size=4, sematic_size=16, beta=8, actions=env.action_space, cosine_cutoff=0.87)


for _ in range(100):    
    action = int(brain.step(observation)[0][0])
    observation, reward, terminated, truncated, info = env.step(action)

    brain.update(observation, reward)

    if terminated or truncated:
        observation, info = env.reset()

env.close()