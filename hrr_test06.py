import numpy as np
import compute.hrr as hrr

next_observation = hrr.projection(np.random.permutation(1*4).reshape(4,1), axis=0)
reward_array = hrr.projection(np.random.permutation(1*4).reshape(4,1), axis=0)
last_action = hrr.projection(np.random.permutation(1*4).reshape(4,1), axis=0)
last_observation = hrr.projection(np.random.permutation(1*4).reshape(4,1), axis=0)




nr = hrr.binding(next_observation, reward_array, axis=0)
nra = hrr.binding(nr, last_action, axis=0)
nrao = hrr.binding(nra, last_observation, axis=0)

# print(action)
# print(ab3)
# print(hrr.cosine_similarity(action, ab3))
# print(hrr.cosine_similarity(action, a2))


# actionObservation = hrr.binding(action, observation, axis=1)
# rewardProbability = hrr.binding(reward, probability, axis=1)

# memory = actionObservation + rewardProbability

# print(memory)
# print(belief)
# print(hrr.unbinding(observation, actionObservation, axis=1))
# print(hrr.unbinding(observation, memory, axis=1))
# print(hrr.cosine_similarity(observation, actionObservation))
# print(hrr.cosine_similarity(observation, memory))
