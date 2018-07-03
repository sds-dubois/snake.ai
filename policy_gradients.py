"""
Reinforcement learning via policy gradients
"""

import random, math, pickle, time
import interface, move, utils
import numpy as np
from agent import Agent
from rl import RLAlgorithm
from collections import defaultdict
from utils import progressBar
from copy import deepcopy
from sklearn.neural_network import MLPRegressor
from constants import NO_MOVE

class PolicyGradientAlgorithm(RLAlgorithm):

    def __init__(self, actions, discount, featureExtractor, exploration = True, weights = None):
        self.actions = actions
        self.n_actions = len(self.actions(interface.State([], [])))
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.exploration = exploration
        self.rl_type = "policy_gradients"

        self.verbose = False
        self.numIters = 0
        self.standardize_rewards = True
        self.step_size = 0.01
        self.buffer_size = 50 # number of rollouts before updating the weights
        self._x_buffer = []
        self._y_buffer = []
        self._p_buffer = []
        self._r_buffer = []

        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.random.rand(self.n_actions, self.featureExtractor.nFeatures()) / np.sqrt(self.featureExtractor.nFeatures())
            # self.weights = np.zeros((self.n_actions, self.featureExtractor.nFeatures()))

    def __str__(self):
        return "PolicyGradients"

    def exportModel(self):
        return self.weights

    def stopExploration(self):
        self.exploration = False

    def evalActions(self, state):
        """ Get the model's confidence to take each action from `state` """
        scores = np.dot(self.weights, self.featureExtractor.arrayExtractor(state, NO_MOVE))
        probas = utils.softmax(scores)
        return probas

    def getActionDetailed(self, state):
        """
        Same as getAction but also return the relative action index.
        """
        self.numIters += 1
        if len(self.actions(state)) == 0:
            return None

        # evaluate model confidence
        probas = self.evalActions(state)

        # choose which (relative) action to take
        if self.exploration:
            action_idx = np.random.choice(range(self.n_actions), p = probas)

            # training on-going, save action taken and proba
            self._x_buffer.append(self.featureExtractor.arrayExtractor(state, NO_MOVE))
            self._y_buffer.append(action_idx)
            self._p_buffer.append(probas)

        else:
            action_idx = np.argmax(probas)

        rel_action = self.actions(state)[action_idx] 
        abs_action = move.Move(self.featureExtractor.toAbsolutePos(state, rel_action.direction()), norm = rel_action.norm())

        if self.verbose:
            # print("")
            # print(state)
            print(self.featureExtractor.dictExtractor(state, NO_MOVE))
            # print(self.featureExtractor.arrayExtractor(state, NO_MOVE))
            print(probas)
            print(rel_action)
            # print(abs_action)

        # rotate relative action to absolute
        return abs_action, action_idx

    def getAction(self, state):
        """
        The strategy implemented by this algorithm.
        If `exploration` is ON, sample action with respect to probas from evalActions.
        """
        a, _ = self.getActionDetailed(state)
        return a

    def getStepSize(self):
        """
        Get the step size to update the weights.
        """
        return self.step_size
        # return 1.0 / math.sqrt(self.numIters)

    def discountedRewards(self, rewards):
        """
        Compute total discounted rewards at each step given the sequence of step rewards.
        """
        n_steps = len(rewards)
        dr = list(np.zeros(n_steps))
        s = 0
        for t in xrange(1, n_steps + 1):
            s = self.discount * s + rewards[n_steps - t]
            dr[n_steps - t] = s

        return dr


    def addRolloutFeedback(self, rewards, rollout_idx):
        self._r_buffer += self.discountedRewards(rewards)

        if ((rollout_idx + 1) % self.buffer_size) == 0:
            # TODO: update weights
            # https://cs231n.github.io/neural-networks-case-study/
            # dL_i / df_k = p_k - 1_(y_i == k)
            # df_k / dX = X.T dot 

            for i, a_idx in enumerate(self._y_buffer):
                self._p_buffer[i][a_idx] -= 1
            
            # gradient with respect to scores
            dlogp = np.asarray(self._p_buffer) # dlogp has shape (n_choices , n_actions)
            # weight by rewards
            if self.standardize_rewards:
                reward_weights = (np.asarray(self._r_buffer) - np.mean(self._r_buffer)) / np.std(self._r_buffer) # standardize rewards (for stability)
            else:
                reward_weights = np.asarray(self._r_buffer)

            weighted_dlogp = (reward_weights * dlogp.T).T # weighted_dlogp has shape (n_choices , n_actions)
    
            X = np.asarray(self._x_buffer) # X has shape (n_choices, n_features)
            dW = np.dot(X.T, weighted_dlogp) # dW has shape (n_features, n_actions)
            w1 = deepcopy(self.weights)

            self.weights -= (self.getStepSize() / math.log(rollout_idx + 2)) * dW.T
            # self.weights -= self.getStepSize() * dW.T
            self._x_buffer, self._y_buffer, self._r_buffer, self._p_buffer = [], [], [], [] # reset buffers
            w2 = deepcopy(self.weights)

            # print("")
            # print(self.weights[:, self.featureExtractor.keyToIndex(("candy", 1, (1,0)))])
            # if rollout_idx > self.buffer_size:
            # raise("End")

