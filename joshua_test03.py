import gymnasium as gym
from compute.brain import Brain
import numpy as np
import math

def update_observation(observation):
    return observation

# f = open("cartpole-joshua.csv", "w")

env = gym.make("CartPole-v1", render_mode="human")
# env = gym.make("CartPole-v1")
observation, info = env.reset()
brain = Brain(episodic_size=4, sematic_size=16, beta=8, actions=env.action_space, euclidean_cutoff=0.5, exploratory=.1)

max_steps = 0
for i in range(100):
    
    terminated = False
    truncated = False
    reward = 0
    steps = 0

    while not terminated and not truncated:
        # print(f'\t{steps}')
        action = int(brain.step(update_observation(observation))[0][0])
        observation, reward, terminated, truncated, info = env.step(action)

        steps = steps + 1
        if terminated:
            reward = -20
        brain.update(update_observation(observation), reward)

    print(f'{i}\t{brain.episodic.memories}\t{brain.sematic.memories}\t{steps}')
    max_steps = max(steps, max_steps)
    observation, info = env.reset()
    # f.write(f'{i}, {steps}\n')
    # f.flush()
print(f'{max_steps}')
# f.close()
env.close()

