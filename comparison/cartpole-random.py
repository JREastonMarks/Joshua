import gymnasium as gym

env = gym.make('CartPole-v1')

def select_action_good(state):
    if state[2]+state[3] < 0:
        return 0
    else:
        return 1
    
num_episodes = 10
num_steps = 500
for episode in range(num_episodes):
    state = env.reset()
    for t in range(1, num_steps+1):
        action = select_action_good(state)
        state, _, done, _ = env.step(action)
        env.render()
        if done:
            break
