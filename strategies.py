"""
Strategies for players.
"""

from utils import *
import random

def greedy(id, state, game):
    """
    Take action which brings us closest to a candy - without even 
    looking at other snakes. 
    """
    actions = game.actions(state, id)
    print actions
    head = state.snakes[id].position[0]
    if len(state.candies) == 0:
        return random.sample(actions, 1)[0]
    best_move = min((dist(add(head, move), candy), move) for candy in state.candies.keys() for move in actions)
    return best_move[1]