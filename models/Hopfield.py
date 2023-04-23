import numpy as np

class Hopfield:
    def __init__(self, size = -1, bias = 0) -> None:
        self.size = size
        self.bias = bias
        self.threshold = 0
        self.weights = None

    def train(self, pattern):

         new_pattern = pattern * np.transpose(pattern)

         if self.weights:
            self.weights = self.weights + new_pattern
         else:
             self.weights = new_pattern

    def query(self, query, maxIterations) -> None:
        energy = self.energy_function(query)

        for iteration in range(maxIterations):
            query = np.sign(self.weights @ query - self.threshold)

            new_energy = self.energy_function(query.reshape(self.size,1))

            if(new_energy == energy):
                 break
            energy = new_energy
        
        return query.reshape()
                  

    def energy_function(self, query):
        queryT = query.transpose()
        part01 = queryT @ self.weights
        part02 = part01 @ query
        part03 = part02[0][0] * -.5
        return part03