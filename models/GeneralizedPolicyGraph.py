import compute.hrr as hrr
import numpy as np
import random

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
        self.root_key = None
        self.populate_graph(observation, self.search_depth)
        if self.root_key is not None:
            self.root = self.graph[self.root_key]
        

    def populate_graph(self, observation, search_depth):
        # If sematic is empty then break
        if self.sematic.weights is None:
            return {}
        # If search depth is less than 1 break
        if search_depth < 1:
            return {}

        # If graph has observation no need to keep going
        observation_key = KeySet(0, observation)

        if self.root_key == None:
            self.root_key = observation_key

        if observation_key in self.graph:
            return

        entry = {}
        # Loop through all actions
        for a in range(self.actions.n):
            action = np.zeros((4, 1))
            action[0][0] = a
            # Build query            
            oa_query = np.append(observation, action)
            oa_query = np.append(oa_query, np.zeros(8))
            oarn = self.sematic.query(oa_query.reshape((16,1)), 10)
            if oarn is None:
                continue
            oarn_split = np.array_split(oarn, 4)

            # Check to see if observation is similar enough
            cosine_similarity = abs(1 - hrr.cosine_similarity(oarn_split[0], observation))
            if(cosine_similarity < self.cosine_cutoff):
                entry[a] = {}
                continue

            # Check to see if action is correct
            if oarn_split[1][0][0] != a:
                entry[a] = {}
                continue

            reward = oarn_split[2]
            next_observation = oarn_split[3]

            # If no result then leave action entry as blank
            if next_observation is None:
                entry[a] = {}
                continue

            # If result then populate entry and go deeper
            entry[a] = {
                'reward': reward[0][0],
                'observation': next_observation
            }
            self.populate_graph(next_observation, search_depth - 1)



        self.graph[observation_key] = entry

    def best_action(self):
        # If graph is empty
        if len(self.graph.keys()) == 0:
            returns = np.zeros((4, 1))
            returns[0][0] = self.actions.sample()
            return returns
        
        # Get keys with actions and rewards
        possible_actions = []
        empty_actions = []
        for action in self.root.keys():
            temp_node = self.root[action]
            if len(temp_node.keys()) == 0:
                empty_actions.append(action)
            else:
                possible_actions.append(action)

        # If empty then return a random action
        if(len(possible_actions) == 0):
            returns = np.zeros((4, 1))
            returns[0][0] = self.actions.sample()
            return returns
        # If an empty exists then check to see if exploration should be done
        if(len(empty_actions) > 0) and (random.random() < self.exploratory):
            returns = np.zeros((4, 1))
            returns[0][0] = random.choice(empty_actions)
            return returns
        
        # If not exploring then calculate the best route (exploitation)
        returns = np.zeros((4, 1))
        action_rewards = {}
        for possible_action in possible_actions:
            max_reward = self.traverse_graph(self.root[possible_action], self.search_depth)
            action_rewards[possible_action] = max_reward

        action_rewards = dict(sorted(action_rewards.items, key=lambda item: item[1]))
        returns[0][0] = list(action_rewards.items)[-1]

        return returns
    
    def traverse_graph(self, node, depth):
        if node is None or len(node.keys()) == 0:
            return 0
        depth = depth - 1
        if depth < 0:
            return 0
        
        observation_key = KeySet(0, node['observation'])
        print(observation_key)
        for key in self.graph.keys():
            print(key)
        new_observation = self.graph[observation_key]

        if len(new_observation.keys()) == 0:
            return 0
        
        rewards = []
        for new_actions in new_observation.keys():
            rewards.append(self.traverse_graph(self.graph[new_actions], depth))

        rewards.sort()

        return rewards[-1] + node['reward']