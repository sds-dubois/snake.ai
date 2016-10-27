__author__ = 'Real_Seb'

from interface import Game
from strategies import greedy, smartGreedy

def controller(strategies, grid_size, candy_ratio=1., max_iter=None):
    game = Game(grid_size, len(strategies), candy_ratio=candy_ratio, max_iter=max_iter)
    state = game.startState()
    while not game.isEnd(state):
        #print state
        state.printGrid(game.grid_size)
        # Compute the actions for each player following its strategy
        actions = {i: strategies[i](i, state, game) for i in state.snakes.keys()}

        # Update the state
        state = game.succ(state, actions)


if __name__ ==  "__main__":
    controller([greedy, smartGreedy, smartGreedy], 20)