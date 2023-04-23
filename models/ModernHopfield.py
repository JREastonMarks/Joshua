import numpy as np
import random

class ModernHopfield:
    def __init__(self, size = -1, bias = 0) -> None:
        self.size = size
        self.bias = bias
        self.threshold = 0
        self.weights = None
        self.memories = 0

    def train(self, pattern):
        
        if(self.memories == 0):
            self.patterns = np.array([pattern])
        else:    
            self.patterns = np.append(self.patterns, [pattern], axis=0)
        self.memories += 1

    def query(self, query, maxIterations):

        for iteration in range(maxIterations):
            update_order = np.arange(0, self.size)
            random.shuffle(update_order)
            for l in update_order:
                new_neurons_positive = query.copy()
                new_neurons_negative = query.copy()

                new_neurons_positive[0][l] = 1
                positive_energy = self.energy_function(new_neurons_positive)
                new_neurons_negative[0][l] = -1
                negative_energy = self.energy_function(new_neurons_negative)

                query[0][l] = np.sign((-1 * positive_energy) + negative_energy)

                # print(f'hit: {iteration }: {l} : {self.energy_function(query)} \t {query}')

        return query.transpose()

    def energy_function(self, query, power=2):
        energy = 0
        query_transpose = query.transpose()
        for pattern_i in range(self.memories):
            pattern = self.patterns[pattern_i]
            part01 = pattern @ query_transpose
            energy += np.power(part01, power)

            # weights_transpose = self.weights.transpose()
            # part01 = weights_transpose @ query
            # part02 = np.power(part01, power)
            # part03 = np.sum(part02)
        return energy * -1
