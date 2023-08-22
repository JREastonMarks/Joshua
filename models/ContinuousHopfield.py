import numpy as np
import compute.hrr as hrr
import cmath
from scipy.special import softmax

class ContinuousHopfield:
    def __init__(self, size = -1, beta = 8) -> None:
        self.size = size
        self.beta = beta
        self.weights = None
        self.memories = 0

    def train(self, pattern):
        self.memories = self.memories + 1
        if self.weights is None:
            self.weights = pattern
        else:
            self.weights = np.append(self.weights, pattern, axis=1)
        

    def query(self, query, max_iterations):
        if(self.weights is None):
            return None
        
        energy = self.energy_function(query)

        for iteration in range(max_iterations):
            query = self.update_rule(query)
            new_energy = self.energy_function(query)

            if(new_energy == energy):
                break
            energy = new_energy
        
        return query
    
    def update_rule(self, query):
        part01 = self.beta * self.weights.transpose() 
        part02 = part01 @ query
        query = self.weights @ softmax(part02)
        return query
    
    def query_or_create(self, query, similarity_cutoff, max_iterations=10):
        result = self.query(query, max_iterations)

        if result is None:
            self.train(query)
            return query

        euclidean_distance = np.linalg.norm(result - query)
        
        if(euclidean_distance > similarity_cutoff):
            self.train(query)
            return query
        
        return result

    def energy_function(self, query):
        if(self.weights is None):
            return None
        part01 = self.weights.transpose() @ query
        part02 = -1 * self.lse(part01)
        part03 = (.5 * (query.transpose() @ query)[0][0]) + part02
        part04 = np.log(self.weights.shape[1])
        part05 = part03 + (pow(self.beta, -1) * part04)
        part06 = part05 + (.5 * pow(self.size, 2))

        return part06

    def lse(self, value):
        beta_pow = pow(self.beta, -1)
        sum = 0
        for a in value:
            holdover = self.beta * a[0]
            holdover_log = cmath.log(holdover)
            sum = sum + holdover_log
        log_sum = cmath.log(sum)
        return beta_pow * log_sum
