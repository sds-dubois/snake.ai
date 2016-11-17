__author__ = 'TheRealSeb'

import utils
import random

class Agent:
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

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all multi-agent searchers.
    Any methods defined here will be available
    to the MinimaxAgent, AlphaBetaAgent & ExpectimaxAgent.
  """

  def __init__(self, evalFn = simpleEvaluationFunction, depth = 2):
    self.evaluationFunction = evalFn
    self.depth = depth

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Minimax agent: the synchronous approach is changed into an asynchronous one
  """

  def getAction(self, mm_agent, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      snake won, snake died, there is no more snake (draw), time is up or there are no legal moves (snake died).

    """
    def vMinMax(state, depth, agent):
      if state.isWin(agent) or state.isLose(agent) or state.isDraw() or state.timesUp() \
              or len(state.actions(agent)) == 0:
        return state.getScore(agent)
      if depth == 0:
        return self.evaluationFunction(state, agent)
      if agent == mm_agent:
        return max(vMinMax(state.generateSuccessor(agent, action), depth-1, state.getNextAgent(agent))
                   for action in state.actions(agent))
      return min(vMinMax(state.generateSuccessor(agent, action), depth, agent+1)
                   for action in state.actions(agent))
    v = []
    agent = gameState.getNextAgent(mm_agent)
    for action in gameState.actions(agent):
      v.append((vMinMax(gameState.generateSuccessor(agent, action), self.depth, 1), action))
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
      if state.isWin(agent) or state.isLose(agent) or state.isDraw() or state.timesUp() \
              or len(state.actions(agent)) == 0:
        return state.getScore(agent)
      if depth == 0:
        return self.evaluationFunction(state, agent)
      if agent == mm_agent:
        v = -float("inf")
        for action in state.actions(agent):
          v = max(v,vMinMax(state.generateSuccessor(agent, action), depth-1, state.getNextAgent(agent), alpha, beta))
          alpha = max(alpha, v)
          if beta <= alpha:
            break
        return v

      v = float("inf")
      for action in state.actions(agent):
        v = min(v, vMinMax(
          state.generateSuccessor(agent, action),
          depth, state.getNextAgent(agent),
          alpha, beta
        ))
        beta = min(beta, v)
        if beta <= alpha:
          break
      return  v
    v = []
    alpha = -float("inf")
    agent = gameState.getNextAgent(mm_agent)
    for action in gameState.actions(agent):
      new_v = vMinMax(gameState.generateSuccessor(agent, action), self.depth, agent, alpha, float("inf"))
      alpha = max(alpha, new_v)
      v.append((new_v, action))
    v_max = max(v)[0]
    return random.sample([a for d, a in v if d == v_max], 1)[0]
