import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")

f = open("cartpole-random.csv", "w")

for i in range(500):
    observation, info = env.reset()
    terminated = False
    truncated = False
    reward = 0
    steps = 0

    while not terminated and not truncated:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        steps = steps + 1

    f.write(f'{i}, {steps}\n')
    f.flush()

f.close()
env.close()
