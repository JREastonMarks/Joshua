import gymnasium as gym
from compute.brain import Brain

env = gym.make("CartPole-v1")
observation, info = env.reset()

brain = Brain(episodic_size=4, sematic_size=16, beta=8, actions=env.action_space, cosine_cutoff=0.43, exploratory=.5)


for _ in range(1000):    
    action = int(brain.step(observation)[0][0])
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        reward = -20
    
    brain.update(observation, reward)
    
    if terminated or truncated:
        observation, info = env.reset()

print(brain.episodic.weights.shape)
print(brain.sematic.weights.shape)
env.close()