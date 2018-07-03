"""
Q learning
"""


import random, math, pickle, time
import interface, utils
import numpy as np
from agent import Agent
from rl import RLAlgorithm
from collections import defaultdict
from utils import progressBar
from copy import deepcopy
from sklearn.neural_network import MLPRegressor

EXPLORATIONPROB = 0.3


class QLearningAlgorithm(RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2, weights = None):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.numIters = 0
        self.rl_type = "qlearning"

        if weights:
            self.weights = defaultdict(float, weights)
        else:
            self.weights = defaultdict(float)

    def __str__(self):
        return "Qlearning"

    def stopExploration(self):
        self.explorationProb = 0

    def exportModel(self):
        return dict(self.weights)

    def evalQ(self, state, action):
        """
        Evaluate Q-function for a given (`state`, `action`)
        """
        score = 0
        for f, v in self.featureExtractor.dictExtractor(state, action):
            score += self.weights[f] * v
        return score


    def getAction(self, state):
        """
        The strategy implemented by this algorithm.
        With probability `explorationProb` take a random action.
        """
        self.numIters += 1
        if len(self.actions(state)) == 0:
            return None

        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.evalQ(state, action), action) for action in self.actions(state))[1]


    def getStepSize(self):
        """
        Get the step size to update the weights.
        """
        return 1.0 / math.sqrt(self.numIters)


    def incorporateFeedback(self, state, action, reward, newState):
        if newState is None:
            return
        
        phi = self.featureExtractor.dictExtractor(state, action)
        pred = sum(self.weights[k] * v for k,v in phi)
        try:
            v_opt = max(self.evalQ(newState, new_a) for new_a in self.actions(newState))
        except:
            v_opt = 0.
        target = reward + self.discount * v_opt
        for k,v in phi:
            self.weights[k] = self.weights[k] - self.getStepSize() * (pred - target) * v


class nnQLearningAlgorithm(QLearningAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2, init_weights = "simple.p", model = None):
        self.actions = actions
        self.discount = discount
        # self.featureExtractor = featureExtractor.arrayExtractor
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.sparse = False
        self.print_time = False
        self.rl_type = "qlearning"

        self.cache_size = 30
        self._reset_cache()

        self.time_feat = []
        self.time_pred = []
        self.time_fit = []

        # TODO
        if model:
            self.mlp = model
            self.numIters = 101 # skip init
        else:
            self.numIters = 0
            with open("data/" + init_weights, "r") as fin:
                init_alg_ = pickle.load(fin)
            self.alg_init = QLearningAlgorithm(actions, discount, featureExtractor, explorationProb, init_alg_.model)
            
            self.mlp = MLPRegressor(
                hidden_layer_sizes = (20,),
                activation = "relu",
                solver = "adam",
                max_iter = 700, #  TODO
                # warm_start TODO
                early_stopping = False,
                verbose = False
            )


    def _reset_cache(self):
        self.cache = 0
        self.x_cache = []
        self.y_cache = []

    def _x_cache(self):
        if self.sparse:
            return self.featureExtractor.sparseMatrixExtractor(self.x_cache)
        else:
            return self.x_cache

    def exportModel(self):
        return self.mlp

    def evalQ(self, state, action):
        """
        Evaluate Q-function for a given (`state`, `action`)
        """
        if self.numIters < 101:
            return self.alg_init.evalQ(state, action)

        if self.sparse:
            return self.mlp.predict(self.featureExtractor.sparseExtractor(self.featureExtractor.dictExtractor(state,action)))[0]
        else:
            return self.mlp.predict([self.featureExtractor.arrayExtractor(state, action)])[0]


    def getAction(self, state):
        """
        The strategy implemented by this algorithm.
        With probability `explorationProb` take a random action.
        """
        self.numIters += 1
        if len(self.actions(state)) == 0:
            return None
        
        if random.random() < self.explorationProb or self.numIters < 102:
            return random.choice(self.actions(state))
        else:
            return max((self.evalQ(state, action), action) for action in self.actions(state))[1]

    def incorporateFeedback(self, state, action, reward, newState):
        if newState is None:
            return
        
        t0 = time.time()
        if self.sparse:
            phi = self.featureExtractor.dictExtractor(state, action)
        else:
            phi = self.featureExtractor.arrayExtractor(state,action)
        t1 = time.time()
        self.time_feat.append(t1-t0)

        if self.numIters < 101:
            pred = self.evalQ(state, action)
        else:       
            if self.sparse:
                pred = self.mlp.predict(self.featureExtractor.sparseExtractor(phi))[0]
            else:
                pred = self.mlp.predict([phi])[0]
            t2 = time.time()
            self.time_pred.append(t2-t1)

        try:
            v_opt = max(self.evalQ(newState, new_a) for new_a in self.actions(newState))
        except:
            v_opt = 0.
        target = reward + self.discount * v_opt

        self.x_cache.append(phi)
        self.y_cache.append(target)
        self.cache += 1

        if self.numIters == 100:
            self.mlp.fit(self._x_cache(), self.y_cache)
            self._reset_cache()

        elif self.numIters > 100 and self.cache == self.cache_size:
            t3 = time.time()
            self.mlp.partial_fit(self._x_cache(), self.y_cache)
            t4 = time.time()
            self.time_fit.append(t4-t3)
            self._reset_cache()

        if self.numIters % 3000 == 0 and self.print_time:
            print "{:.2f}\t{:.2f}\t{:.2f}".format(1000. * np.mean(self.time_feat), 1000. * np.mean(self.time_pred), 1000. * np.mean(self.time_fit))




# class QLambdaLearningAlgorithm(QLearningAlgorithm):
#
#     # TODO: update train method to deal with new game interface
#
#     def __init__(self, actions, discount, featureExtractor, explorationProb=0.2, lambda_ = 0.2, weights = None):
#         self.actions = actions
#         self.discount = discount
#         self.featureExtractor = featureExtractor.dictExtractor
#         self.explorationProb = explorationProb
#         self.numIters = 0
#         self.lambda_ = lambda_

#         if weights:
#             with open("data/" + weights, "rb") as fin:
#                 weights_ = pickle.load(fin)
#                 self.weights = defaultdict(float, weights_)
#         else:
#             self.weights = defaultdict(float)

#     def getAction(self, state):
#         """
#         The strategy implemented by this algorithm.
#         With probability `explorationProb` take a random action.
#         Return `action, is_optimal_action`.
#         """
#         self.numIters += 1
#         if len(self.actions(state)) == 0:
#             return None, True
        
#         if random.random() < self.explorationProb:
#             return random.choice(self.actions(state)), False
#         else:
#             return max((self.evalQ(state, action), action) for action in self.actions(state))[1], True

#     def updateWeights(self, phi, delta):
#         for k,v in phi:
#             self.weights[k] = self.weights[k] - self.getStepSize() * delta * v

#     def incorporateFeedback(self, state, action, reward, newState, history = []):
#         if newState is None:
#             return
        
#         phi = self.featureExtractor(state, action)

#         pred = sum(self.weights[k] * v for k,v in phi)
#         try:
#             v_opt = max(self.evalQ(newState, new_a) for new_a in self.actions(newState))
#         except:
#             v_opt = 0.
#         target = reward + self.discount * v_opt
#         delta = pred - target
#         for k,v in phi:
#             self.weights[k] = self.weights[k] - self.getStepSize() * delta * v

#         for i in xrange(len(history)):
#             delta *= self.discount * self.lambda_
#             self.updateWeights(history[-(i+1)], delta)

#     def train(self, strategies, grid_size, num_trials = 100, max_iter = 1000, verbose = False):
#         print "RL training"
#         totalRewards = []  # The rewards we get on each trial
#         rl_id = len(strategies)
#         for trial in xrange(num_trials):
#             progressBar(trial, num_trials)
#             game = interface.Game(grid_size, len(strategies) + 1, candy_ratio = 1., max_iter = max_iter)
#             state = game.startState()
#             totalDiscount = 1
#             totalReward = 0
#             points = state.snakes[rl_id].points
#             history = []
#             while not game.isEnd(state) and rl_id in state.snakes:
#                 # Compute the actions for each player following its strategy
#                 actions = {i: strategies[i](i, state) for i in state.snakes.keys() if i != rl_id}
#                 action, optimal_action = self.getAction(state)
#                 actions[rl_id] = action

#                 newState = game.succ(state, actions)
#                 if rl_id in newState.snakes:
#                     reward = newState.snakes[rl_id].points - points
#                     if len(newState.snakes) == 1: # it won
#                         reward += 10.
#                     points = newState.snakes[rl_id].points
#                     self.incorporateFeedback(state, action, reward, newState, history)
#                 else: # it died
#                     reward = - 10.
#                     self.incorporateFeedback(state, action, reward, newState, history)

#                 # add decsion to history, or reset if non-greedy choice
#                 if optimal_action:
#                     history.append(self.featureExtractor(state, action))
#                 else:
#                     history = []

#                 totalReward += totalDiscount * reward
#                 totalDiscount *= self.discount
#                 state = newState

#             if verbose:
#                 print "Trial %d (totalReward = %s)" % (trial, totalReward)
#             totalRewards.append(totalReward)

#         progressBar(num_trials, num_trials)
#         print "Average reward:", sum(totalRewards)/num_trials
#         return totalRewards
