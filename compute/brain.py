from models.ContinuousHopfield import ContinuousHopfield
from models.GeneralizedPolicyGraph import GeneralizedPolicyGraph
import compute.hrr as hrr
import numpy as np

class Brain:
    def __init__(self, episodic_size = 4, sematic_size = 16, beta = 8, actions = None, cosine_cutoff = 0.01, exploratory=0.1, search_depth=6) -> None:
        self.episodic = ContinuousHopfield(episodic_size, beta=beta)
        self.sematic = ContinuousHopfield(sematic_size, beta=beta)
        self.episodic_size = episodic_size
        self.sematic_size = sematic_size
        self.actions = actions
        self.cosine_cutoff = float(cosine_cutoff)
        self.exploratory = exploratory
        self.search_depth = search_depth

        self.gpg = GeneralizedPolicyGraph(self.episodic, self.sematic, actions, exploratory, search_depth, float(cosine_cutoff))
        self.last_action = None
        self.last_observation = None
        

    def step(self, observation):
        observation = observation.reshape((self.episodic_size, 1))        
        episodic_observation = self.episodic.query_or_create(observation, self.cosine_cutoff, max_iterations=10)
        
        # Create Generalized Policy Graph
        self.gpg.create(episodic_observation)

        # Get best action from graph
        action = self.gpg.best_action()

        self.last_observation = episodic_observation
        self.last_action = action

        return action

    def update(self, observation, reward):
        observation = observation.reshape((self.episodic_size, 1))
        next_observation = self.episodic.query_or_create(observation, self.cosine_cutoff, max_iterations=10)

        reward_array = np.zeros((2, 1))
        reward_array[0][0] = reward
        
        oa = np.append(self.last_observation, self.last_action)
        oar = np.append(oa, reward_array)
        oarn = np.append(oar, next_observation)
        oarn.resize(self.sematic_size, 1)

        self.sematic.query_or_create(oarn, self.cosine_cutoff, max_iterations=10)
        

    def init_brain(self, observation):
        self.episodic.train(observation)
        return observation