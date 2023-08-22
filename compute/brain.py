from models.ContinuousHopfield import ContinuousHopfield
from models.GeneralizedPolicyGraph import GeneralizedPolicyGraph
import compute.hrr as hrr
import numpy as np

class Brain:
    def __init__(self, episodic_size = 4, sematic_size = 16, beta = 8, actions = None, euclidean_cutoff = 0.4, exploratory=0.1, search_depth=6) -> None:
        self.episodic = ContinuousHopfield(episodic_size, beta=beta)
        self.sematic = ContinuousHopfield(sematic_size, beta=beta)
        self.episodic_size = episodic_size
        self.sematic_size = sematic_size
        self.actions = actions
        self.euclidean_cutoff = float(euclidean_cutoff)
        self.exploratory = exploratory
        self.search_depth = search_depth

        self.gpg = GeneralizedPolicyGraph(self.episodic, self.sematic, actions, exploratory, search_depth, float(euclidean_cutoff))
        self.last_action = None
        self.last_observation = None
        

    def step(self, observation):
        observation = observation.reshape((self.episodic_size, 1))        

        # Does the episodic memory have this memory or one similar enough?
        episodic_observation = self.episodic.query_or_create(observation, self.euclidean_cutoff, max_iterations=10)

        
        action = np.zeros((2, 1))

        # If new then return random action
        if(np.array_equal(episodic_observation, observation)):
            action[0][0] = self.actions.sample()
        else:
            # If not new then create GPG
            self.gpg.create(episodic_observation)
            action = self.gpg.best_action()
        
        self.last_observation = episodic_observation
        self.last_action = action
        

        return action

    def update(self, observation, reward):
        # Create OARN
        next_observation = observation.reshape((self.episodic_size, 1))

        reward_array = np.zeros((2, 1))
        reward_array[0][0] = reward

        oa = np.append(self.last_observation, self.last_action)
        oar = np.append(oa, reward_array)
        oarn = np.append(oar, next_observation)
        oarn.resize(self.sematic_size, 1)

        result = self.sematic.query(oarn, 10)

        if result is None:
            self.sematic.train(oarn)
            return
        
        euclidean_distance = np.linalg.norm(result - oarn)

        if(euclidean_distance > self.euclidean_cutoff):
            self.sematic.train(oarn)

        