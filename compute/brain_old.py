from models.ContinuousHopfield import ContinuousHopfield
import compute.hrr as hrr
import numpy as np

class Brain:
    def __init__(self, episodic_size = 4, sematic_size = 16, beta = 8, default_action = None, cosine_cutoff = 0.01,) -> None:
        self.episodic = ContinuousHopfield(episodic_size, beta=beta)
        self.sematic = ContinuousHopfield(sematic_size, beta=beta)
        self.episodic_size = episodic_size
        self.sematic_size = sematic_size
        self.default_action = np.zeros((1, episodic_size))
        self.default_action[0][0] = default_action
        self.cosine_cutoff = cosine_cutoff
        self.last_belief = None
        self.last_action = None
        self.last_observation = None
        

    def step(self, observation):
        observation = np.reshape(observation, (self.episodic_size, 1))
        self.last_observation = observation
        
        # Get observation from episodic memory
        episodic_observation = self.episodic.query(observation, 10)        
        # Is observation less than the cosine_cutoff
        if(episodic_observation is None):
            self.episodic.train(observation)
            episodic_observation = observation
        else:
            cosine_similarity = abs(1 - hrr.cosine_similarity(episodic_observation, observation))
            print(f'Cosine Similarity: {cosine_similarity}')
            if(cosine_similarity > self.cosine_cutoff):
                self.episodic.train(observation)
        
        episodic_observation_project = hrr.projection(episodic_observation, axis=0)
        # Retrieve belief from somatic memory
        belief = self.sematic.query(episodic_observation, 10)
        if(belief is None):
            action_projection = hrr.projection(self.default_action, axis=1)
            belief = hrr.binding(episodic_observation_project, action_projection, axis=0)
            self.sematic.train(belief)

        # Get action from belief
        action = hrr.unbinding(belief, episodic_observation_project, axis=0)

        # Get action from episodic memory

        self.last_action = action
        self.last_belief = belief
        return action

    def update(self, observation, reward):

        pass