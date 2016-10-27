"""
Strategies for players.
"""

from utils import *

def greedy(id, state, game):
    """
    Take action which brings us closest to a candy - without even 
    looking at other snakes. 
    """
    actions = game.actions(state, id)
    head = state.snakes[id].position[0]
    best_move = min((dist(add(head, move), candy), move) for candy in state.candies.keys() for move in actions)
    return best_move[1]