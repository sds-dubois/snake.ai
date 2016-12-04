__author__ = 'TheRealSeb'

import utils
import random
import numpy as np

class Agent(object):
    """
    An agent must define a getAction method
    """
    def __init__(self, index=0):
        self.index = index

    def getAction(self, state):
        """
        The Agent will receive a GameState and
        must return an move from Move (Direction and Norm)
        """
        raise NotImplementedError("getAction not implemented")

def simpleEvaluationFunction(state, agent):
    """
        This default evaluation function just returns the score of the state.
        The score is the same one displayed in the Pacman GUI.

        This evaluation function is meant for use with adversarial search agents
        (not reflex agents).
    """
    return state.getScore(agent)

def greedyEvaluationFunction(state, agent):
    return state.getScore(agent) -min(
        float(utils.dist(state.snakes[agent].head(), candy))/(2*state.grid_size) for candy in state.candies.iterkeys()
    )

def cowardDepthFunction(state, mm_agent):
    return 1

class MultiAgentSearchAgent(Agent):
    """
        This class provides some common elements to all multi-agent searchers.
        Any methods defined here will be available
        to the MinimaxAgent, AlphaBetaAgent & ExpectimaxAgent.
    """

    def __init__(self, evalFn = simpleEvaluationFunction, depth = lambda s, a: 2):
        self.evaluationFunction = evalFn
        self.depth = depth

class FunmaxAgent(MultiAgentSearchAgent):
    """
        Minimax agent: the synchronous approach is changed into an asynchronous one
    """

    def __init__(self, func, evalFn = simpleEvaluationFunction, depth = lambda s, a: 2):
        super(FunmaxAgent, self).__init__(evalFn=evalFn, depth=depth)
        self.func = func

    def getAction(self, mm_agent, gameState):
        """
            Returns the minimax action from the current gameState using self.depth
            and self.evaluationFunction. Terminal states can be found by one of the following:
            snake won, snake died, there is no more snake (draw), time is up or there are no legal moves (snake died).

        """
        def vMinMax(state, depth, agent):

            # Edge cases
            if state.isWin(mm_agent) or state.isLose(mm_agent) or state.isDraw():
                return state.getScore(mm_agent), None
            if len(state.actions(agent)) == 0 and agent == mm_agent:
                return -float("inf"), None
            if len(state.actions(agent)) == 0:
                return vMinMax(state, depth, state.getNextAgent(agent))
            if depth == 0:
                return self.evaluationFunction(state, mm_agent), None

            # Max case
            if agent == mm_agent:
                return max((vMinMax(state.generateSuccessor(agent, action), depth-1, state.getNextAgent(agent))[0], action)
                                     for action in state.actions(agent))
            # Other case (func)
            return self.func(vMinMax(state.generateSuccessor(agent, action), depth, state.getNextAgent(agent))
                                     for action in state.actions(agent))
        v = []

        if self.depth(gameState, mm_agent) <= 0:
            return self.evaluationFunction(gameState, mm_agent)

        agent = gameState.getNextAgent(mm_agent)
        while(len(gameState.actions(agent)) == 0 and agent != mm_agent):
            agent = gameState.getNextAgent(agent)

        if agent == mm_agent:
            return random.sample(gameState.actions(mm_agent), 1)[0]

        for action in gameState.actions(agent):
            v.append(vMinMax(gameState.generateSuccessor(agent, action),
                             self.depth(gameState, mm_agent), gameState.getNextAgent(agent)))
        v_min = min(v)[0]
        return random.sample([a for d, a in v if d == v_min], 1)[0]
    
class MinimaxAgent(FunmaxAgent):
    def __init__(self, evalFn = simpleEvaluationFunction, depth = lambda s: 2):
        super(MinimaxAgent, self).__init__(min, evalFn=evalFn, depth=depth)


class ExpectimaxAgent(MultiAgentSearchAgent):
    def getAction(self, mm_agent, gameState):
        """
            Returns the minimax action from the current gameState using self.depth
            and self.evaluationFunction. Terminal states can be found by one of the following:
            snake won, snake died, there is no more snake (draw), time is up or there are no legal moves (snake died).

        """
        def vMinMax(state, depth, agent):

            # Edge cases
            if state.isWin(mm_agent) or state.isLose(mm_agent) or state.isDraw():
                return state.getScore(mm_agent)
            if len(state.actions(agent)) == 0 and agent == mm_agent:
                return -float("inf")
            if len(state.actions(agent)) == 0:
                return vMinMax(state, depth, state.getNextAgent(agent))
            if depth == 1 and agent == mm_agent:
                return self.evaluationFunction(state, mm_agent)

            # Max case
            M = -float("inf")
            if agent == mm_agent:
                for action in state.actions(agent):
                    changes = state.generateSuccessor(agent, action)
                    M = max(M, vMinMax(state, depth-1, state.getNextAgent(agent)))
                    state.reverseChanges(changes)
                return M
            # Mean case
            avg = 0.
            for action in state.actions(agent):
                changes = state.generateSuccessor(agent, action)
                avg += vMinMax(state, depth, state.getNextAgent(agent))
                state.reverseChanges(changes)
            return float(avg)/len(state.actions(agent))
        v = []
        if len(gameState.actions(mm_agent)) == 0:
            return None

        for action in gameState.actions(mm_agent):
            changes = gameState.generateSuccessor(mm_agent, action)
            v.append((vMinMax(gameState,self.depth(gameState, mm_agent), gameState.getNextAgent(mm_agent)), action))
            gameState.reverseChanges(changes)
        v_max = max(v)[0]
        return random.sample([a for d, a in v if d == v_max], 1)[0]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
        Your minimax agent with alpha-beta pruning
    """

    def getAction(self, mm_agent, gameState):
        """
            Returns the minimax action using self.depth and self.evaluationFunction
        """

        # BEGIN_YOUR_CODE (our solution is 49 lines of code, but don't worry if you deviate from this)
        def vMinMax(state, depth, agent, alpha, beta):
            if state.isWin(mm_agent) or state.isLose(mm_agent) or state.isDraw():
                return state.getScore(mm_agent), None
            if len(state.actions(agent)) == 0 and agent == mm_agent:
                return -float("inf"), None
            if len(state.actions(agent)) == 0:
                return vMinMax(state, depth, state.getNextAgent(agent), alpha, beta)
            if depth == 0:
                return self.evaluationFunction(state, mm_agent), None
            if agent == mm_agent:
                v = (-float("inf"),None)
                for action in state.actions(agent):
                    changes = state.generateSuccessor(agent, action)
                    vs = vMinMax(state, depth-1, state.getNextAgent(agent), alpha, beta)
                    state.reverseChanges(changes)
                    if (vs[0] > v[0]) or (vs[0] == v[0] and bool(random.getrandbits(1))):
                        v = (vs[0],action)
                    alpha = max(alpha, v[0])
                    if beta <= alpha:
                        break
                return v

            v = (float("inf"), None)
            for action in state.actions(agent):
                changes = state.generateSuccessor(agent, action)
                vs = vMinMax(state,depth, state.getNextAgent(agent), alpha, beta)
                state.reverseChanges(changes)
                if (vs[0] < v[0]) or (vs[0] == v[0] and bool(random.getrandbits(1))):
                    v = vs
                beta = min(beta, v[0])
                if beta <= alpha:
                    break
            return v

        if self.depth(gameState, mm_agent) <= 0:
            return self.evaluationFunction(gameState, mm_agent)

        v = []
        beta = float("inf")
        agent = gameState.getNextAgent(mm_agent)
        while(len(gameState.actions(agent)) == 0 and agent != mm_agent):
            agent = gameState.getNextAgent(agent)

        if agent == mm_agent:
            if len(gameState.actions(mm_agent)) == 0:
                return None
            return random.sample(gameState.actions(mm_agent), 1)[0]

        for action in gameState.actions(agent):
            changes = gameState.generateSuccessor(agent, action)
            new_v, best_action = vMinMax(gameState, self.depth(gameState, mm_agent),
                                                                     gameState.getNextAgent(agent), -float("inf"), beta)
            gameState.reverseChanges(changes)
            beta = min(beta, new_v)
            v.append((new_v, best_action))
        v_min = min(v)[0]
        if len([a for d,a in v if d == v_min and a is not None]) == 0:
            if gameState.actions(mm_agent) == []:
                return None
            return random.sample(gameState.actions(mm_agent), 1)[0]
        return random.sample([a for d, a in v if d == v_min and a is not None], 1)[0]

