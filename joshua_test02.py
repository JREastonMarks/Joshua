import gymnasium as gym
from compute.brain import Brain

f = open("cartpole-joshua.csv", "w")

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()
brain = Brain(episodic_size=4, sematic_size=16, beta=8, actions=env.action_space, cosine_cutoff=0.43, exploratory=.5)

for i in range(500):
    
    terminated = False
    truncated = False
    reward = 0
    steps = 0

    while not terminated and not truncated:
        action = int(brain.step(observation)[0][0])
        observation, reward, terminated, truncated, info = env.step(action)
        steps = steps + 1
        if truncated:
            reward = -20
        brain.update(observation, reward)

    observation, info = env.reset()
    f.write(f'{i}, {steps}\n')
    f.flush()

f.close()
env.close()