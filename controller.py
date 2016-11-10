import sys
from interface import Game
from strategies import randomStrategy, greedyStrategy, smartGreedyStrategy

def controller(strategies, grid_size, candy_ratio = 1., max_iter = None, verbose = 1):
    game = Game(grid_size, len(strategies), candy_ratio = candy_ratio, max_iter = max_iter)
    state = game.startState()
    while not game.isEnd(state):
        # Print state
        if verbose > 0:
            state.printGrid(game.grid_size)

        # Compute the actions for each player following its strategy
        actions = {i: strategies[i](i, state, game) for i in state.snakes.keys()}

        if verbose > 1:
            print state
            print actions

        # Update the state
        state = game.succ(state, actions)

    if verbose > 0:
        state.printGrid(game.grid_size)

    return state


if __name__ ==  "__main__":
    if len(sys.argv) > 1:
        max_iter = int(sys.argv[1])
    else:
        max_iter = None
    #controller([randomStrategy, greedyStrategy, smartGreedyStrategy], 20, max_iter = max_iter, verbose = 1)
    controller([smartGreedyStrategy, smartGreedyStrategy], 20, max_iter = max_iter, verbose = 1)