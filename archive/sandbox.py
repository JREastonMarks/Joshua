import numpy as np
import random

rowMemories = 5
columnMemories = 5
memories = rowMemories * columnMemories
neurons = np.ones((rowMemories, columnMemories))
update_order = np.arange(0, memories)
random.shuffle(update_order)

l = update_order[0]

# print(neurons)
# print(update_order)
changeRow = int(l / rowMemories)
changeCol = l % rowMemories
print(l)
print(changeRow)
print(changeCol)
print(neurons[changeRow][changeCol])