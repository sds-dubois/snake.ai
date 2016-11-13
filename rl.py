"""
Reinforcement learning
"""

import random, math
import interface, utils
from collections import defaultdict
from utils import progressBar

class QLearningAlgorithm:
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    def evalQ(self, state, action):
        """
        Evaluate Q-function for a given (`state`, `action`)
        """
        score = 0
        for f, v in self.featureExtractor(state, action):
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
        
        phi = self.featureExtractor(state, action)
        pred = sum(self.weights[k] * v for k,v in phi)
        try:
            v_opt = max(self.evalQ(newState, new_a) for new_a in self.actions(newState))
        except:
            v_opt = 0.
        target = reward + self.discount * v_opt
        for k,v in phi:
            self.weights[k] = self.weights[k] - self.getStepSize() * (pred - target) * v


############################################################

def simpleFeatureExtractor(state, action, id):
    head = state.snakes[id].position[0]
    features = [(('candy', utils.add(head, c, mu = -1), v), 1.) for c,v in state.candies.iteritems()]
    features += [(('adv', utils.add(head, s.position[0], mu = -1), v), 1.) for k,s in state.snakes.iteritems() if k != id]
    features += [(('x', head[0]), 1.), (('y', head[1]), 1.)]
    return features

def simpleFeatureExtractor2(state, action, id):
    head = state.snakes[id].position[0]
    features = [(('candy', utils.add(head, c, mu = -1), v), 1.) for c,v in state.candies.iteritems()]
    features += [(('adv-head', utils.add(head, s.position[0], mu = -1), v), 1.) for k,s in state.snakes.iteritems() if k != id]
    features += [(('adv-tail', utils.add(head, t, mu = -1), v), 1.) for k,s in state.snakes.iteritems() for t in s.position[1:] if k != id and utils.dist(head, t) < 5]
    features += [(('my-tail', utils.add(head, t, mu = -1), v), 1.) for t in state.snakes[id].position[1:] if utils.dist(head, t) < 5]
    features += [(('x', head[0]), 1.), (('y', head[1]), 1.)]
    return features

############################################################

def train(rl, strategies, grid_size, candy_ratio = 1., num_trials=100, max_iter=1000, verbose=False):
    print "RL training"
    totalRewards = []  # The rewards we get on each trial
    rl_id = len(strategies)
    for trial in xrange(num_trials):
        progressBar(trial, num_trials)
        game = interface.Game(grid_size, len(strategies) + 1, candy_ratio = candy_ratio, max_iter = max_iter)
        state = game.startState()
        totalDiscount = 1
        totalReward = 0
        points = state.snakes[rl_id].points
        while not game.isEnd(state) and rl_id in state.snakes:
            # Compute the actions for each player following its strategy
            actions = {i: strategies[i](i, state, game) for i in state.snakes.keys() if i != rl_id}
            action = rl.getAction(state)
            actions[rl_id] = action

            newState = game.succ(state, actions)
            if rl_id in newState.snakes:
                reward = newState.snakes[rl_id].points - points
                points = newState.snakes[rl_id].points
                rl.incorporateFeedback(state, action, reward, newState)
            else:
                rl.incorporateFeedback(state, action, 0, None)

            totalReward += totalDiscount * reward
            totalDiscount *= rl.discount
            state = newState

        if verbose:
            print "Trial %d (totalReward = %s)" % (trial, totalReward)
        totalRewards.append(totalReward)

    progressBar(num_trials, num_trials)
    print "Average reward:", sum(totalRewards)/num_trials
    return totalRewards

def rl_strategy(strategies, featureExtractor, grid_size, candy_ratio = 1., num_trials=100, max_iter=1000, verbose=False):
    rl_id = len(strategies)
    actions = lambda s : s.simple_actions(rl_id)
    features = lambda s,a : simpleFeatureExtractor(s, a, rl_id)
    rl = QLearningAlgorithm(actions, discount = 1.0, featureExtractor = features, explorationProb = 0.2)
    train(rl, strategies, grid_size, num_trials=num_trials, max_iter=max_iter, verbose=verbose)
    rl.explorationProb = 0
    strategy = lambda id,s,g : rl.getAction(s)
    return strategy