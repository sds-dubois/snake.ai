import sys
from interface import Game
from strategies import randomStrategy, greedyStrategy, smartGreedyStrategy, opportunistStrategy
from minimax import MinimaxAgent, AlphaBetaAgent, greedyEvaluationFunction
from rl import rl_strategy, load_rl_strategy, simpleFeatureExtractor0, simpleFeatureExtractor1, simpleFeatureExtractor2

def controller(strategies, grid_size, candy_ratio = 1., max_iter = None, verbose = 1):
    game = Game(grid_size, len(strategies), candy_ratio = candy_ratio, max_iter = max_iter)
    state = game.startState()
    while not game.isEnd(state):
        # Print state
        if verbose > 0:
            state.printGrid(game.grid_size)

        # Compute the actions for each player following its strategy
        actions = {i: strategies[i](i, state) for i in state.snakes.iterkeys()}

        if verbose > 1:
            print state
            print actions

        # Update the state
        state = game.succ(state, actions, copy = False)

    if verbose > 0:
        state.printGrid(game.grid_size)

    return state


if __name__ ==  "__main__":
    if len(sys.argv) > 1:
        max_iter = int(sys.argv[1])
    else:
        max_iter = None

    minimax_agent = MinimaxAgent(depth=2)
    alphabeta_agent = AlphaBetaAgent(depth=1, evalFn= greedyEvaluationFunction)
    controller([randomStrategy, opportunistStrategy, alphabeta_agent.getAction],
               20, max_iter = max_iter, verbose = 1)

    #rlStrategy = load_rl_strategy("weights.p", [randomStrategy, smartGreedyStrategy, opportunistStrategy], simpleFeatureExtractor1)
    #strategies = [randomStrategy, smartGreedyStrategy, opportunistStrategy, rlStrategy]
    #controller(strategies, 20, max_iter = max_iter, verbose = 1)