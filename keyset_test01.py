import numpy as np

class KeySet(object):
    def __init__(self, i, arr):
        self.i = i
        self.arr = arr
        
    def __hash__(self):
        return hash(np.array_str(self.arr))
        # return hash((self.i, hash(self.arr.tobytes())))


action_first = np.zeros((4, 1))
action_first[0][0] = 1
# action_first_keyset = action_first.flatten() #KeySet(0, action_first)
# action_first_keyset = KeySet(0, action_first)
action_first_keyset = np.array_str(action_first)

action_second = np.zeros((4, 1))
action_second[0][0] = 2
# action_second_keyset = KeySet(0, action_second)
action_second_keyset = np.array_str(action_second)


action_third = np.zeros((4, 1))
action_third[0][0] = 1
# action_third_keyset = KeySet(0, action_third)
action_third_keyset = np.array_str(action_third)

print(action_first_keyset)
print(action_first.transpose())
print(action_second_keyset)
print(action_second.transpose())
print(action_third_keyset)

action_dict = {}
action_dict[action_first_keyset] = 'test_a'
action_dict[action_second_keyset] = 'test_b'

print(action_dict)
print(action_dict[action_first_keyset])
print(action_dict[action_third_keyset])