"""
Evolutionary Strategy

References:
- Augmented random search https://arxiv.org/abs/1803.07055
"""

import random, math, pickle, time
import interface, move, hp, utils
import numpy as np
from collections import defaultdict
from utils import progressBar
from copy import deepcopy
from sklearn.neural_network import MLPRegressor
from constants import NO_MOVE


class EvolutionaryAlgorithm:

    def __init__(self, actions, discount, featureExtractor, strategies, grid_size, max_game_iter, sigma=0.1, alpha=0.1, mode=0, population_size=10, filename = None):
        self.actions = actions
        self.n_actions = len(self.actions(interface.State([], [])))
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.strategies = strategies # oponents
        self.grid_size = grid_size
        self.max_game_iter = max_game_iter
        self.sigma = sigma
        self.alpha = alpha
        # self.numIters = 0

        self.population_size = population_size if (mode != 0) else (2*population_size)
        self.n_candidate_rollouts = 10
        self.weights = np.random.rand(self.n_actions, self.featureExtractor.nFeatures()) / np.sqrt(self.featureExtractor.nFeatures())
        # self.weights = np.zeros(self.featureExtractor.n_features())
        self.mode = mode     # mode: 0 = default, 1 = brs, 2 = ars

        if filename:
            init_hp = hp.load_from(filename)
            self.weights = init_hp.model

    def exportModel(self):
        return self.weights

    def _sample_candidates(self):
        return [self.weights + (self.sigma * np.random.randn(*self.weights.shape)) for _ in xrange(self.population_size)]

    def _evaluate_avg(self, weight_candidate, n_rollouts):
        return np.mean([self._evaluate(weight_candidate) for _ in xrange(n_rollouts)])

    def _evaluate(self, weight_candidate):
        agent = ESAgent(self.actions, self.discount, self.featureExtractor, weights = weight_candidate).getAgent()

        agents = deepcopy(self.strategies) # add current agent to strategies
        agents.append(agent)

        game = interface.Game(grid_size, len(agents), candy_ratio = 1., max_iter = max_iter)
        game.start(agents)
        totalDiscount = 1
        totalReward = 0

        while not game.isEnd() and agent.isAlive(game):
            # Compute the actions for each player following its strategy
            actions = game.agentActions()
            newState = game.succ(game.current_state, actions)

            reward = agent.lastReward(game)

            totalReward += totalDiscount * reward
            totalDiscount *= self.discount

        # fitness = totalReward
        fitness = game.current_state.currentScore(agent_id)
        # fitness = max(0, totalReward)  # we need positive values
        return fitness

    def _update_weights(self):
        candidates = self._sample_candidates()
        if self.mode > 0:
            candidates += [-w for w in candidates]

        raw_returns = np.asarray([self._evaluate_avg(candidate, self.n_candidate_rollouts) for candidate in candidates])
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
        # self.numIters = 0
        
        if filename:
            with open("data/" + filename, "rb") as fin:
                self.weights = pickle.load(fin)
        elif weights is not None:
            self.weights = weights
        else:
            raise ValueError("ES agent initialized without weights")


    def evalActions(self, state):
        """ Get the model's confidence to take each action from `state` """
        scores = np.dot(self.weights, self.featureExtractor.arrayExtractor(state, NO_MOVE))
        probas = utils.softmax(scores)
        return probas


    def getAction(self, state):
        """
        The strategy implemented by this algorithm.
        """
        if len(self.actions(state)) == 0:
            return None
    
        # evaluate model confidence
        probas = self.evalActions(state)
        action_idx = np.argmax(probas)

        rel_action = self.actions(state)[action_idx] 
        abs_action = move.Move(self.featureExtractor.toAbsolutePos(state, rel_action.direction()), norm = rel_action.norm())
        return abs_action

    def getAgent(self):
        agent = Agent(name = "ES", strategy = (lambda i,s : self.getAction(s)))
        return agent
        




############################################################

def es_strategy(strategies, featureExtractor, discount, grid_size, num_trials = 100, max_iter = 1000, filename = "es/weights1.p", verbose = False):
    es_id = len(strategies)
    actions = lambda s : s.all_rel_actions(es_id)

    # ea = EvolutionaryAlgorithm(actions, discount, featureExtractor, strategies, grid_size, max_game_iter = max_iter, sigma = 0.0001, alpha = 0.0001, mode = 1, population_size = 20, filename = None)
    ea = EvolutionaryAlgorithm(actions, discount, featureExtractor, strategies, grid_size, max_game_iter = max_iter, sigma = 0.001, alpha = 0.001, mode = 1, population_size = 20, filename = "pg-linear-r6-1000.p")
    # (self, actions, discount, featureExtractor, strategies, grid_size, max_game_iter, sigma=0.1, alpha=0.1)
    ea.train(num_trials)

    agent = ESAgent(ea.actions, ea.discount, ea.featureExtractor, weights = ea.weights).getAgent()

    # save learned weights
    with open("data/" + filename, "wb") as fout:
        pickle.dump(ea.weights, fout)
    
    with open("info/{}txt".format(filename[:-1]), "wb") as fout:
        print >> fout, "strategies: ", [s.__str__() for s in strategies]
        print >> fout, "feature radius: ", featureExtractor.radius
        print >> fout, "grid: {}, trials: {}, max_iter: {}".format(grid_size, num_trials, max_iter)
        print >> fout, "discount: {}".format(discount)
    
    return agent


def load_es_strategy(filename, strategies, featureExtractor, discount):
    es_id = len(strategies)
    actions = lambda s : s.all_rel_actions(es_id)

    agent = ESAgent(actions, discount, featureExtractor, weights = None, filename = filename).getAgent()
    return agent
