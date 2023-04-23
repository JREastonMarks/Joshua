import numpy as np
import compute.hrr as hrr


action = hrr.projection(np.random.permutation(4*4).reshape(1,16), axis=1)
observation = hrr.projection(np.random.permutation(4*4).reshape(1,16), axis=1)
reward = hrr.projection(np.random.permutation(4*4).reshape(1,16), axis=1)
probability = hrr.projection(np.random.permutation(4*4).reshape(1,16), axis=1)

axis = 1

ao2 = hrr.binding(action, observation, axis=1)
a2 = hrr.unbinding(ao2, observation, axis=1)

aor3 = hrr.binding3(action, observation, reward, axis=1)
ab3 = hrr.unbinding(aor3, observation, axis=1)

# print(action)
# print(ab3)
print(hrr.cosine_similarity(action, ab3))
print(hrr.cosine_similarity(action, a2))


# actionObservation = hrr.binding(action, observation, axis=1)
# rewardProbability = hrr.binding(reward, probability, axis=1)

# memory = actionObservation + rewardProbability

# print(memory)
# print(belief)
# print(hrr.unbinding(observation, actionObservation, axis=1))
# print(hrr.unbinding(observation, memory, axis=1))
# print(hrr.cosine_similarity(observation, actionObservation))
# print(hrr.cosine_similarity(observation, memory))
