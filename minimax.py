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

def greedyEvaluationFunction(state, agent):
  return state.getScore(agent) -min(
    float(utils.dist(state.snakes[agent].head(), candy))/(2*state.grid_size) for candy in state.candies.iterkeys()
  )

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
      if state.isWin(mm_agent) or state.isLose(mm_agent) or state.isDraw():
#        print "putain"
        return state.getScore(mm_agent), None
      if len(state.actions(agent)) == 0 and agent == mm_agent:
#        print "encule"
        return -float("inf"), None
      if len(state.actions(agent)) == 0:
#        print "sa mere"
        return vMinMax(state, depth, state.getNextAgent(agent))
      if depth == 0:
#        print "la hyene"
        return self.evaluationFunction(state, mm_agent), None
      if agent == mm_agent:
#        print "python de la fournaise"
        return max((vMinMax(state.generateSuccessor(agent, action), depth-1, state.getNextAgent(agent))[0], action)
                   for action in state.actions(agent))
#      print "Chibre a sa maman"
      return min(vMinMax(state.generateSuccessor(agent, action), depth, state.getNextAgent(agent))
                   for action in state.actions(agent))
    v = []

    agent = gameState.getNextAgent(mm_agent)
    while(len(gameState.actions(agent)) == 0 and agent != mm_agent):
      agent = gameState.getNextAgent(agent)

    if agent == mm_agent:
      return random.sample(gameState.actions(mm_agent), 1)[0]

    for action in gameState.actions(agent):
      v.append(vMinMax(gameState.generateSuccessor(agent, action), self.depth, gameState.getNextAgent(agent)))
    v_min = min(v)[0]
    return random.sample([a for d, a in v if d == v_min], 1)[0]

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
          vs = vMinMax(state.generateSuccessor(agent, action),
                           depth-1, state.getNextAgent(agent), alpha, beta)
          if (vs[0] > v[0]) or (vs[0] == v[0] and bool(random.getrandbits(1))):
            v = (vs[0],action)
          alpha = max(alpha, v[0])
          if beta <= alpha:
            break
        return v

      v = (float("inf"), None)
      for action in state.actions(agent):
        vs = vMinMax(
          state.generateSuccessor(agent, action),
          depth, state.getNextAgent(agent),
          alpha, beta
        )
        if (vs[0] < v[0]) or (vs[0] == v[0] and bool(random.getrandbits(1))):
          v = vs
        beta = min(beta, v[0])
        if beta <= alpha:
          break
      return v
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
      new_v, best_action = vMinMax(gameState.generateSuccessor(agent, action), self.depth,
                                   gameState.getNextAgent(agent), -float("inf"), beta)
      beta = min(beta, new_v)
      v.append((new_v, best_action))
    v_min =  min(v)[0]
    if len([a for d,a in v if d == v_min and a is not None]) == 0:
      return random.sample(gameState.actions(agent), 1)[0]
    return random.sample([a for d, a in v if d == v_min and a is not None], 1)[0]
