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
        self.cosine_cutoff = cosine_cutoff
        self.exploratory = exploratory
        self.search_depth = search_depth

        self.gpg = GeneralizedPolicyGraph(self.episodic, self.sematic, actions, exploratory, search_depth, cosine_cutoff)
        self.last_action = None
        self.last_observation = None
        

    def step(self, observation):
        observation = observation.reshape((self.episodic_size, 1))
        # Get observation
        episodic_observation = self.episodic.query(observation, 10)

        # If observation can not be found
        if episodic_observation is None:
            self.episodic.train(observation)
            episodic_observation = observation
        else:
            # Check to see if retrieved observation and observation are similar enough
            cosine_similarity = abs(1 - hrr.cosine_similarity(episodic_observation, observation))
            if(cosine_similarity > self.cosine_cutoff):
                self.episodic.train(observation)
                episodic_observation = observation
        
        # Create Generalized Policy Graph
        self.gpg.create(observation)

        # Get best action from graph
        action = self.gpg.best_action()

        self.last_observation = episodic_observation
        self.last_action = action

        return action

    def update(self, observation, reward):
        reward_array = np.zeros((4, 1))
        reward_array[0][0] = reward
        
        # last_observation * last_action + reward + observation
        # nr = hrr.binding(hrr.projection(observation, axis=0), hrr.projection(reward_array, axis=0), axis=0)
        # nra = hrr.binding(nr, hrr.projection(self.last_action, axis=0), axis=0)
        # nrao = hrr.binding(nra, hrr.projection(self.last_observation, axis=0), axis=0)
        oa = np.append(self.last_observation, self.last_action)
        oar = np.append(oa, reward_array)
        oarn = np.append(oar, observation)
        # nrao = np.append(observation, [reward_array, self.last_action, self.last_observation])
        sematic_belief = self.sematic.query(oarn, 10)

        if sematic_belief is None:
            self.sematic.train(oarn.reshape((16,1)))
        else:
            cosine_similarity = abs(1 - hrr.cosine_similarity(sematic_belief, oarn))
            if(cosine_similarity > self.cosine_cutoff):
                self.episodic.train(oarn.reshape((16,1)))
                # sematic_belief = nrao

    def init_brain(self, observation):
        self.episodic.train(observation)
        return observation