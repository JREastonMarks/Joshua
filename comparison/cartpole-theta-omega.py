# https://towardsdatascience.com/how-to-beat-the-cartpole-game-in-5-lines-5ab4e738c93f
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")

f = open("cartpole-theta-omega.csv", "w")

def theta_omega_policy(obs):
    theta, w = obs[2:4]
    if abs(theta) < 0.03:
        return 0 if w < 0 else 1
    else:
        return 0 if theta < 0 else 1
    
for i in range(500):
    observation, info = env.reset()
    terminated = False
    truncated = False
    reward = 0
    steps = 0

    while not terminated and not truncated:
        action = theta_omega_policy(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        steps = steps + 1

    f.write(f'{i}, {steps}\n')

f.close()
env.close()
