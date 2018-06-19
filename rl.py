
import random, math, pickle, time
import interface, utils
import numpy as np
from agent import Agent
from utils import progressBar
from copy import deepcopy

class RLAlgorithm(object):

    def __str__(self):
        return "RLAlgorithm"

    def getAction(self, state):
        raise NotImplementedError("getAction not implemented")

    # def getAgent(self):
    #     raise NotImplementedError("getAction not implemented")

    def getAgent(self, stopExploration = False):
        if stopExploration:
            self.stopExploration()
        agent = Agent(name = self.__str__(), strategy = (lambda i,s : self.getAction(s)))
        return agent

    def exportModel(self):
        raise NotImplementedError("exportModel not implemented")

    def stopExploration(self):
        raise NotImplementedError("stopExploration not implemented")

    def incorporateFeedback(self, state, action, reward, newState):
        raise NotImplementedError("incorporateFeedback not implemented")

    def addRolloutFeedback(self, rewards, rollout_idx):
        raise NotImplementedError("addRolloutFeedback not implemented")

    # def train(self):
    #     raise NotImplementedError("train not implemented")

    def train(self, opponents, grid_size, num_trials=100, max_iter=1000, verbose=False):
        print "RL training"
        totalRewards = []  # The rewards we get on each trial
        # rl_id = len(opponents)
        rl_agent = self.getAgent()

        agents = deepcopy(opponents) # add current agent to strategies
        agents.append(rl_agent)

        for trial in xrange(num_trials):
            # game = interface.Game(grid_size, len(strategies) + 1, candy_ratio = 1., max_iter = max_iter)
            # state = game.startState()
            game = interface.Game(grid_size, len(agents), candy_ratio = 1., max_iter = max_iter)
            game.start(agents)
            totalDiscount = 1
            totalReward = 0
            rewards = []

            while not game.isEnd() and rl_agent.isAlive(game):
                # Compute the actions for each player following its strategy
                # actions = {i: strategies[i](i, state) for i in state.snakes.keys() if i != rl_id}
                # action = self.getAction(state)
                # actions[rl_id] = action
                actions = game.agentActions()
                newState = game.succ(game.current_state, actions)

                reward = rl_agent.lastReward(game)
                rewards.append(reward)

                totalReward += totalDiscount * reward
                totalDiscount *= self.discount

                if self.rl_type == "qlearning":
                    self.incorporateFeedback(game.previous_state, actions[rl_agent.getPlayerId()], reward, game.current_state)

            if self.rl_type == "policy_gradients":
                self.addRolloutFeedback(rewards, trial)

            progressBar(trial, num_trials, info = "Last reward: {}".format(totalReward))
            if verbose:
                print "Trial %d (totalReward = %s)" % (trial, totalReward)
            totalRewards.append(totalReward)

        progressBar(num_trials, num_trials)
        print "Average reward:", sum(totalRewards)/num_trials
        return totalRewards