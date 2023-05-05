import numpy as np
import compute.hrr as hrr

observation_left_final = np.array([-2.4, -0.05, -.2095, -0.05])
observation_left_middle = np.array([-1.225, -0.05, -0.129, -0.05])
observation_left_start = np.array([-0.05, -0.05, -0.05, -0.05])

observation_right_start = np.array([0.05, 0.05, 0.05, 0.05])
observation_right_middle = np.array([1.225, 0.05, 0.129, 0.05])
observation_right_final = np.array([2.4, 0.05, .2095, 0.05])

print(observation_left_final)

left_final_middle = hrr.cosine_similarity(observation_left_final, observation_left_middle)
left_middle_start = hrr.cosine_similarity(observation_left_middle, observation_left_start)

left_right_start = hrr.cosine_similarity(observation_left_start, observation_right_start)

right_middle_start = hrr.cosine_similarity(observation_right_start, observation_right_middle)
right_final_middle = hrr.cosine_similarity(observation_right_middle, observation_right_final)

print(left_final_middle)
print(left_middle_start)
print(left_right_start)
print(right_middle_start)
print(right_final_middle)

total = sum([left_final_middle, left_middle_start, left_right_start, right_middle_start, right_final_middle])
print(total)
print(total / 5)
