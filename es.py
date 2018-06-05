"""
Evolutionary Strategy

References:
- Augmented random search https://arxiv.org/abs/1803.07055
"""

import random, math, pickle, time
import interface, utils
import numpy as np
from collections import defaultdict
from utils import progressBar
from copy import deepcopy
from sklearn.neural_network import MLPRegressor


class EvolutionaryAlgorithm:

    def __init__(self, actions, discount, featureExtractor, strategies, grid_size, max_game_iter, sigma=0.1, alpha=0.1, mode=0, population_size=10, filename = None):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.strategies = strategies # oponents
        self.grid_size = grid_size
        self.max_game_iter = max_game_iter
        self.sigma = sigma
        self.alpha = alpha
        # self.numIters = 0
        self.population_size = population_size if (mode != 0) else (2*population_size)  
        self.weights = np.zeros(self.featureExtractor.n_features())
        self.mode = mode
        # mode: 0 = default, 1 = brs, 2 = ars

        if filename:
            with open("data/" + filename, "r") as fin:
                self.weights = pickle.load(fin)
                # weights_ = pickle.load(fin)
                # for k,v in weights_.iteritems():
                #     self.weights[featureExtractor.keyToIndex(k)] = v

    def _sample_candidates(self):
        return [self.weights + (self.sigma * np.random.randn(*self.weights.shape)) for _ in xrange(self.population_size)]


    def _evaluate_avg(self, weight_candidate, n_rollouts = 10):
        return np.mean([self._evaluate(weight_candidate) for _ in xrange(n_rollouts)])

    def _evaluate(self, weight_candidate):
        agent = ESAgent(self.actions, self.discount, self.featureExtractor, weights = weight_candidate)
        agent_id = len(self.strategies)
        game = interface.Game(self.grid_size, len(self.strategies) + 1, candy_ratio = 1., max_iter = self.max_game_iter)
        state = game.startState()
        totalDiscount = 1
        totalReward = 0
        points = state.snakes[agent_id].points
        while not game.isEnd(state) and agent_id in state.snakes:
            # Compute the actions for each player following its strategy
            actions = {i: self.strategies[i](i, state) for i in state.snakes.keys() if i != agent_id}
            action = agent.getAction(state)
            actions[agent_id] = action

            newState = game.succ(state, actions)
            if agent_id in newState.snakes:
                reward = newState.snakes[agent_id].points - points
                if len(newState.snakes) == 1: # it won
                    reward += 10.
                points = newState.snakes[agent_id].points
            else: # it died
                reward = - 10.

            totalReward += totalDiscount * reward
            totalDiscount *= self.discount
            state = newState

        fitness = totalReward
        # fitness = state.currentScore(agent_id)
        # fitness = max(0, totalReward)  # we need positive values
        return fitness

    def _update_weights(self):
        candidates = self._sample_candidates()
        if self.mode > 0:
            candidates += [-w for w in candidates]

        raw_returns = np.asarray([self._evaluate_avg(candidate, 5) for candidate in candidates])
        returns = (raw_returns - np.mean(raw_returns)) / np.std(raw_returns)

        if self.mode == 2:
            scaling_coef = self.alpha / (len(candidates) * np.std(returns))
        else:
            scaling_coef = self.alpha / (len(candidates))

        self.weights += scaling_coef * np.sum([r_i * (w_i - self.weights) for (r_i,w_i) in zip(returns, candidates)])
        return np.mean(raw_returns)


    def train(self, num_trials):
        for trial in xrange(num_trials):
            mr = self._update_weights()
            progressBar(trial, num_trials, info = "Mean returns: {}".format(mr))
        progressBar(num_trials, num_trials)
        print "Done"


class ESAgent:

    def __init__(self, actions, discount, featureExtractor, weights = None, filename = None):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.numIters = 0
        
        if filename:
            with open("data/" + filename, "rb") as fin:
                self.weights = pickle.load(fin)
        elif weights is not None:
            self.weights = weights
        else:
            raise ValueError("ES agent initialized without weights")
                

    def evalQ(self, state, action):
        """
        Evaluate Q-function for a given (`state`, `action`)
        """
        # return self.mlp.predict([self.featureExtractor.arrayExtractor(state, action)])[0]
        return self.weights.dot(self.featureExtractor.arrayExtractor(state, action))

    def getAction(self, state):
        """
        The *deterministic* strategy implemented by this algorithm.
        """
        self.numIters += 1
        if len(self.actions(state)) == 0:
            return None
        
        return max((self.evalQ(state, action), action) for action in self.actions(state))[1]




############################################################

def es_strategy(strategies, featureExtractor, discount, grid_size, num_trials = 100, max_iter = 1000, filename = "es/weights1.p", verbose = False):
    es_id = len(strategies)
    actions = lambda s : s.simple_actions(es_id)

    ea = EvolutionaryAlgorithm(actions, discount, featureExtractor, strategies, grid_size, max_game_iter = max_iter, sigma = 0.0001, alpha = 0.0001, mode = 1, population_size = 20, filename = "es-simple.p")
    # (self, actions, discount, featureExtractor, strategies, grid_size, max_game_iter, sigma=0.1, alpha=0.1)
    ea.train(num_trials)

    agent = ESAgent(ea.actions, ea.discount, ea.featureExtractor, weights = ea.weights)
    strategy = lambda id,s : agent.getAction(s)


    # save learned weights
    with open("data/" + filename, "wb") as fout:
        pickle.dump(ea.weights, fout)
    
    with open("info/{}txt".format(filename[:-1]), "wb") as fout:
        print >> fout, "strategies: ", [s.__name__ for s in strategies]
        print >> fout, "feature radius: ", featureExtractor.radius
        print >> fout, "grid: {}, trials: {}, max_iter: {}".format(grid_size, num_trials, max_iter)
        print >> fout, "discount: {}".format(discount)
    
    return strategy


def load_es_strategy(filename, strategies, featureExtractor, discount):
    es_id = len(strategies)
    actions = lambda s : s.simple_actions(es_id)

    agent = ESAgent(actions, discount, featureExtractor, weights = None, filename = filename)
    strategy = lambda id,s : agent.getAction(s)
    return strategy
