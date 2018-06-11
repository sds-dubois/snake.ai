"""
Reinforcement learning via policy gradients
"""

import random, math, pickle, time
import interface, move, utils
import numpy as np
from collections import defaultdict
from utils import progressBar
from copy import deepcopy
from sklearn.neural_network import MLPRegressor
from constants import NO_MOVE


class PolicyGradientAlgorithm:

    def __init__(self, actions, discount, featureExtractor, exploration = True, weights = None):
        self.actions = actions
        self.n_actions = len(self.actions(interface.State([], [])))
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.exploration = exploration

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

    def export_model(self):
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

    def train(self, strategies, grid_size, num_trials = 100, max_iter = 1000, verbose = False):
        print "RL training"
        totalRewards = []  # The rewards we get on each trial
        rl_id = len(strategies)
        for trial in xrange(num_trials):
            progressBar(trial, num_trials)
            game = interface.Game(grid_size, len(strategies) + 1, candy_ratio = 1., max_iter = max_iter)
            state = game.startState()
            totalDiscount = 1
            totalReward = 0
            rewards = []
            points = state.snakes[rl_id].points
            while not game.isEnd(state) and rl_id in state.snakes:
                # Compute the actions for each player following its strategy
                actions = {i: strategies[i](i, state) for i in state.snakes.keys() if i != rl_id}
                action = self.getAction(state)
                actions[rl_id] = action

                newState = game.succ(state, actions)
                if rl_id in newState.snakes:
                    reward = newState.snakes[rl_id].points - points
                    if len(newState.snakes) == 1: # it won
                        reward += 10.
                    points = newState.snakes[rl_id].points
                else: # it died
                    reward = - 10.

                rewards.append(reward)
                totalReward += totalDiscount * reward
                totalDiscount *= self.discount
                state = newState
            self.addRolloutFeedback(rewards, trial)

            if verbose:
                print "Trial %d (totalReward = %s)" % (trial, totalReward)
            totalRewards.append(totalReward)

        progressBar(num_trials, num_trials)
        print "Average reward:", sum(totalRewards)/num_trials
        return totalRewards


############################################################

# # def rl_strategy(strategies, featureExtractor, discount, grid_size, q_type = "linear", lambda_ = None, num_trials = 100, max_iter = 1000, filename = "weights.p", verbose = False):
def pg_strategy(strategies, featureExtractor, game_hp, rl_hp, num_trials = 100, filename = "weights.p", verbose = False):
    rl_id = len(strategies)
    actions = lambda s : s.all_rel_actions(rl_id)

    rl = PolicyGradientAlgorithm(actions, discount = game_hp.discount, featureExtractor = featureExtractor, exploration = True)
    rl.train(strategies, game_hp.grid_size, num_trials = num_trials, max_iter = game_hp.max_iter, verbose = verbose)
    rl.stopExploration()

    strategy = lambda id,s : rl.getAction(s)

    rl_hp.save_model(rl.export_model(), filename)
    
    with open("info/{}txt".format(filename[:-1]), "wb") as fout:
        print >> fout, "strategies: ", [s.__name__ for s in strategies]
        print >> fout, "feature radius: ", rl_hp.radius
        print >> fout, "grid: {}, lambda: {}, trials: {}, max_iter: {}".format(game_hp.grid_size, rl_hp.lambda_, num_trials, game_hp.max_iter)
        print >> fout, "discount: {}, fiter actions: {}".format(game_hp.discount, rl_hp.filter_actions)
    
    return strategy

# # def load_rl_strategy(filename, strategies, featureExtractor, discount, q_type = "linear"):
def load_pg_strategy(rl_hp, strategies, featureExtractor):
    rl_id = len(strategies)
    actions = lambda s : s.all_rel_actions(rl_id)

    rl = PolicyGradientAlgorithm(actions, discount = None, featureExtractor = featureExtractor, exploration = False, weights = rl_hp.model)
    strategy = lambda id,s : rl.getAction(s)
    return strategy
