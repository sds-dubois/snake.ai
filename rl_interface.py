"""
Reinforcement learning
"""

import random, math, pickle, time
import interface, utils
import numpy as np
from agent import Agent
from qlearning import QLearningAlgorithm, nnQLearningAlgorithm
from policy_gradients import PolicyGradientAlgorithm
from collections import defaultdict
from utils import progressBar
from copy import deepcopy
from sklearn.neural_network import MLPRegressor

QL_EXPLORATIONPROB = 0.3


def rl_strategy(strategies, featureExtractor, game_hp, rl_hp, num_trials = 100, filename = "weights.p", verbose = False):

    rl_id = len(strategies)

    if rl_hp.rl_type == "policy_gradients":
        actions = lambda s : s.all_rel_actions(rl_id)
    elif rl_hp.rl_type == "qlearning" and rl_hp.filter_actions:
        actions = lambda s : s.simple_actions(rl_id)
    elif rl_hp.rl_type == "qlearning":
        actions = lambda s : s.all_actions(rl_id)
    else:
        raise("rl_type error")


    if rl_hp.rl_type == "policy_gradients":
        rl = PolicyGradientAlgorithm(actions, discount = game_hp.discount, featureExtractor = featureExtractor, exploration = True)

    else:
        if rl_hp.lambda_:
            if rl_hp.q_type != "linear":
                print "Warning, linear model with eligibility traces instead of", rl_hp.q_type
            rl = QLambdaLearningAlgorithm(actions, discount = game_hp.discount, featureExtractor = featureExtractor, lambda_ = rl_hp.lambda_, explorationProb = QL_EXPLORATIONPROB)
        elif rl_hp.q_type == "nn":
            rl = nnQLearningAlgorithm(actions, discount = game_hp.discount, featureExtractor = featureExtractor, explorationProb = QL_EXPLORATIONPROB, init_weights = "simple-ql-r6.p")
        else:
            rl = QLearningAlgorithm(actions, discount = game_hp.discount, featureExtractor = featureExtractor, explorationProb = QL_EXPLORATIONPROB)


    rl.train(strategies, game_hp.grid_size, num_trials = num_trials, max_iter = game_hp.max_iter, verbose = verbose)
    rl_agent = rl.getAgent(stopExploration = True)

    rl_hp.save_model(rl.exportModel(), filename)
    
    with open("info/{}txt".format(filename[:-1]), "wb") as fout:
        print >> fout, "strategies: ", [s.__str__() for s in strategies]
        print >> fout, "feature radius: ", rl_hp.radius
        print >> fout, "grid: {}, lambda: {}, trials: {}, max_iter: {}".format(game_hp.grid_size, rl_hp.lambda_, num_trials, game_hp.max_iter)
        print >> fout, "discount: {}, fiter actions: {}, explorationProb: {}".format(game_hp.discount, rl_hp.filter_actions, QL_EXPLORATIONPROB)
    
    return rl_agent

def load_rl_strategy(rl_hp, strategies, featureExtractor):
    rl_id = len(strategies)

    if rl_hp.rl_type == "policy_gradients":
        actions = lambda s : s.all_rel_actions(rl_id)
    elif rl_hp.rl_type == "qlearning" and rl_hp.filter_actions:
        actions = lambda s : s.simple_actions(rl_id)
    elif rl_hp.rl_type == "qlearning":
        actions = lambda s : s.all_actions(rl_id)
    else:
        raise("rl_type error")


    if rl_hp.rl_type == "policy_gradients":
        rl = PolicyGradientAlgorithm(actions, discount = None, featureExtractor = featureExtractor, exploration = False, weights = rl_hp.model)
    elif rl_hp.q_type == "nn":
        rl = nnQLearningAlgorithm(actions, discount = None, featureExtractor = featureExtractor, explorationProb = 0, model = rl_hp.model)
    elif rl_hp.q_type == "linear":
        rl = QLearningAlgorithm(actions, discount = None, featureExtractor = featureExtractor, explorationProb = 0, weights = rl_hp.model)
    
    rl_agent = rl.getAgent(stopExploration = True)
    return rl_agent
