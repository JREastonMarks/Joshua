import gymnasium as gym
from compute.brain import Brain
import numpy as np

def update_observation(observation):
    returns = observation[2:]
    returns[0] = np.rad2deg(returns[0])
    return returns

# f = open("cartpole-joshua.csv", "w")

env = gym.make("CartPole-v1", render_mode="human")
# env = gym.make("CartPole-v1")
observation, info = env.reset()
brain = Brain(episodic_size=2, sematic_size=8, beta=8, actions=env.action_space, cosine_cutoff=0.9, exploratory=.5)

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

