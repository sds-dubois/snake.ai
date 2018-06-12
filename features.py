"""
Feature extractor for Reinforcement Learning
"""

import utils
import numpy as np
from copy import deepcopy
from scipy.sparse import csr_matrix
from constants import CANDY_VAL, CANDY_BONUS

class FeatureExtractor:
    def __init__(self, id_, grid_size, radius_ = 16):
        self.id = id_
        self.grid_size = grid_size
        self.radius = radius_
        self.rotate = True

        tiles = self.radius**2 + (self.radius - 1)**2
        self.prefix = {
            "candy1" : 0,
            "candy2" : tiles,
            "adv-head" : 2 * tiles,
            "adv-tail" : 3 * tiles,
            "my-tail" : 4 * tiles,
            "wall-xr" : 5 * tiles,
            "wall-xl" : 5 * tiles + self.grid_size,
            "wall-yt" : 5 * tiles + 2 * self.grid_size,
            "wall-yb" : 5 * tiles + 3 * self.grid_size,
            "non-auth": 5 * tiles + 4 * self.grid_size,
            "tot" : 1 + 1 + 5 * tiles + 4 * self.grid_size
        }
        # self.prefix = {
        #     "candy1" : 0,
        #     "candy2" : tiles,
        #     "adv-head" : 2 * tiles,
        #     "adv-tail" : 3 * tiles,
        #     "my-tail" : 4 * tiles,
        #     "x" : 5 * tiles,
        #     "y" : 5 * tiles + (self.grid_size - 1)/2,
        #     "non-auth": 5 * tiles + 2 * int((self.grid_size - 1)/2),
        #     "tot" : 1 + 1 + 5 * tiles + 2 * int((self.grid_size - 1)/2)
        # }

        self.index = {}
        i = 0
        for x in xrange(1-self.radius, self.radius):
            for y in xrange(1-self.radius, self.radius):
                if utils.dist((0,0), (x,y)) < self.radius:
                    self.index[(x,y)] = i
                    i += 1


    def nFeatures(self):
        return self.prefix["tot"]

    def toAbsolutePos(self, state, p):
        agent = state.snakes[self.id]
        dir_ = agent.orientation()
        return utils.rotateBack(p, dir_)
                
    # def n_features(self):
    #     return self.prefix["tot"]
    
    def relativePos(self, ref, p, orientation):
        if self.rotate:
            return utils.rotate(utils.add(ref, p, mu = -1), orientation)
        else:
            return utils.add(ref, p, mu = -1)

    def dictExtractor(self, state, action):
        if action is None:
            return [('trapped', 1.)]


        if action.norm() == 1:
            agent = state.snakes[self.id]
            # pretend agent moves with action
            authorized_move = agent.authorizedMove(action) # check before moving
            last_tail = agent.position.pop()
            agent.position.appendleft(utils.add(agent.head(), action.direction()))
        elif action.norm() > 1:
            agent = deepcopy(state.snakes[self.id])
            authorized_move = agent.authorizedMove(action) # check before moving
            agent.move(action)
        else:
            agent = state.snakes[self.id]
            authorized_move = True

        head = agent.head()
        dir_ = agent.orientation()
        def relPos(p):
            return self.relativePos(head, p, dir_)

        features = [
            (('candy', v, relPos(c)), 1.) 
                for c,v in state.candies.iteritems() 
                if utils.dist(head, c) < self.radius
        ]
        features += [
            (('adv-head', relPos(s.head())), 1.) 
                for k,s in state.snakes.iteritems() 
                if k != self.id and utils.dist(head, s.head()) < self.radius
        ]
        features += [
            (('adv-tail', relPos(s.position[i])), 1.) 
                for k,s in state.snakes.iteritems() 
                for i in xrange(1, len(s.position)) 
                if k != self.id and utils.dist(head, s.position[i]) < self.radius
        ]
        features += [
            (('my-tail', relPos(state.snakes[self.id].position[i])), 1.) 
                for i in xrange(1, len(state.snakes[self.id].position)) 
                if utils.dist(head, state.snakes[self.id].position[i]) < self.radius
        ]

        # features += [
        #     (('x', min(head[0], state.grid_size - 1 - head[0])), 1.), 
        #     (('y', min(head[1], state.grid_size - 1 - head[1])), 1.)
        # ]
        if dir_ == (0,1):
            features += [
                (('wall-xl', head[0]), 1.), 
                (('wall-xr', state.grid_size - 1 - head[0]), 1.), 
                (('wall-yt', state.grid_size - 1 - head[1]), 1.),
                (('wall-yb', head[1]), 1.)
            ]
        elif dir_ == (0,-1):
            features += [
                (('wall-xr', head[0]), 1.), 
                (('wall-xl', state.grid_size - 1 - head[0]), 1.), 
                (('wall-yb', state.grid_size - 1 - head[1]), 1.),
                (('wall-yt', head[1]), 1.)
            ]
        elif dir_ == (1,0):
            features += [
                (('wall-yb', head[0]), 1.), 
                (('wall-yt', state.grid_size - 1 - head[0]), 1.), 
                (('wall-xl', state.grid_size - 1 - head[1]), 1.),
                (('wall-xr', head[1]), 1.)
            ]
        elif dir_ == (-1,0):
            features += [
                (('wall-yt', head[0]), 1.), 
                (('wall-yb', state.grid_size - 1 - head[0]), 1.), 
                (('wall-xr', state.grid_size - 1 - head[1]), 1.),
                (('wall-xl', head[1]), 1.)
            ]

        if not authorized_move:
            features += [("non-auth", 1.)]

        # revert changes
        if action.norm() == 1:
            agent.position.popleft()
            agent.position.append(last_tail)

        return features
    

    def arrayExtractor(self, state, action):
        features = self.dictExtractor(state, action)
        arrayFeatures = np.zeros(self.prefix["tot"])

        for f,v in features:
            if f == "trapped":
                arrayFeatures[self.prefix["tot"] - 1] += 1.
            elif f == "non-auth":
                arrayFeatures[self.prefix["non-auth"]] += 1.
            elif f[0] == "candy" and f[1] == CANDY_VAL:
                arrayFeatures[self.prefix["candy1"] + self.index[f[2]]] += 1.
            elif f[0] == "candy" and f[1] == CANDY_BONUS:
                arrayFeatures[self.prefix["candy2"] + self.index[f[2]]] += 1.
            elif f[0] in ["adv-head", "adv-tail", "my-tail"]:
                arrayFeatures[self.prefix[f[0]] + self.index[f[1]]] += 1.
            elif f[0] in ["wall-xr", "wall-xl", "wall-yt", "wall-yb"]: #["x", "y"]:
                arrayFeatures[self.prefix[f[0]] + f[1]] += 1.
            else:
                print "ERROR: feature not recognized", f
        return arrayFeatures

    def keyToIndex(self, f):
        if f == "trapped":
            return self.prefix["tot"] - 1
        elif f == "non-auth":
            return self.prefix["non-auth"]
        elif f[0] == "candy" and f[1] == CANDY_VAL:
            return self.prefix["candy1"] + self.index[f[2]]
        elif f[0] == "candy" and f[1] == CANDY_BONUS:
            return self.prefix["candy2"] + self.index[f[2]]
        elif f[0] in ["adv-head", "adv-tail", "my-tail"]:
            return self.prefix[f[0]] + self.index[f[1]]
        elif f[0] in ["wall-xr", "wall-xl", "wall-yt", "wall-yb"]: #["x", "y"]:
            return self.prefix[f[0]] + f[1]

    def sparseExtractor(self, features):
        return csr_matrix((np.ones(len(features)), [self.keyToIndex(f) for f,v in features], [0, len(features)]), shape = (1, self.prefix["tot"]))
    
    def sparseMatrixExtractor(self, feature_list):
        idx = [0]
        count = 0
        cols = []
        for features in feature_list:
            count += len(features)
            idx.append(count)
            cols += [self.keyToIndex(f) for f,v in features]
        return csr_matrix((np.ones(len(cols)), cols, idx), shape = (len(feature_list), self.prefix["tot"]))
