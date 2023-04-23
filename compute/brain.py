from models.ContinuousHopfield import ContinuousHopfield
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

        self.last_action = None
        self.last_observation = None
        

    def step(self, observation):
        # Get observation
        episodic_observation = self.episodic.query(observation, 10)
        # If observation can not be found
        if episodic_observation is None:
            self.episodic.train(observation)
            episodic_observation = observation
        else:
            cosine_similarity = abs(1 - hrr.cosine_similarity(episodic_observation, observation))
            if(cosine_similarity > self.cosine_cutoff):
                self.episodic.train(observation)
                episodic_observation = observation
        
        # Create Generalized Policy Graph
        gpg = create_policy_graph(episodic_observation)

        # Get best action from graph
        action = gpg.best_action()

        self.last_observation = episodic_observation
        self.last_action = action
        
        pass

    def update(self, observation, reward):
        # last_observation * last_action + reward + observation
        oa = hrr.binding(self.last_observation, self.last_action)
        oar = hrr.binding(oa, reward)
        belief = hrr.binding(oar, observation)

        pass

    def init_brain(self, observation):
        self.episodic.train(observation)
        return observation
    
    def create_policy_graph(self, observation):
        graph = GeneralizedPolicyGraph()

        return graph