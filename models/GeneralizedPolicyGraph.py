import compute.hrr as hrr
import numpy as np

class KeySet(object):
    def __init__(self, i, arr):
        self.i = i
        self.arr = arr
        
    def __hash__(self):
        return hash((self.i, hash(self.arr.tostring())))

class GeneralizedPolicyGraph:
    def __init__(self, episodic, sematic, actions, exploratory,search_depth, cosine_cutoff):
        self.graph = {}
        self.episodic = episodic
        self.sematic = sematic
        self.actions = actions
        self.exploratory = exploratory
        self.search_depth = search_depth
        self.cosine_cutoff = cosine_cutoff

    def create(self, observation):
        self.graph = {}
        self.populate_graph(observation, self.search_depth)

    def populate_graph(self, observation, search_depth):
        # If sematic is empty then break
        if self.sematic.weights is None:
            return
        # If search depth is less than 1 break
        if search_depth < 1:
            return

        # If graph has observation no need to keep going
        observation_key = KeySet(0, observation)
        if observation_key in self.graph:
            return

        entry = {}
        # Loop through all actions
        for a in range(self.actions.n):
            # Build query
            # oa_query = hrr.binding(observation, a, axis=1)
            oa_query = np.append(observation, a)
            oa_query = np.append(oa_query, np.zeros(11))
            oarn = self.sematic.query(oa_query.reshape((16,1)), 10)

            # arn = hrr.unbinding(oarn, observation, axis=1)
            # rn = hrr.unbinding(arn, a, axis=1)
            oarn_split = np.array_split(oarn, 4)
            reward = oarn_split[2]
            next_observation = oarn_split[3]

            # Check for result
            # next_observation = self.episodic(rn, 10)
            # cosine_similarity = abs(1 - hrr.cosine_similarity(observation, next_observation))
            # if(cosine_similarity > self.cosine_cutoff):
            #     continue
            
            
            # If no result then leave action entry as blank
            if next_observation is not None:
                entry[a] = {}
                continue
            # reward = hrr.unbinding(rn, next_observation, axis=1)

            # If result then populate entry and go deeper
            entry[a] = {
                'reward': reward,
                'observation': next_observation
            }
            self.populate_graph(next_observation, search_depth - 1)



        self.graph[observation_key] = entry

    def best_action(self):
        if len(self.graph.keys()) == 0:
            returns = np.zeros((1, 4))
            returns[0][0] = self.actions.sample()
            return returns
        return self.actions.sample()