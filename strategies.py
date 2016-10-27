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
    head = state.snakes[id].position[0]
    if len(state.candies) == 0:
        return random.sample(actions, 1)[0]
    best_move = min((dist(add(head, move), candy), move) for candy in state.candies.keys() for move in actions)
    return best_move[1]

def smartGreedy(id, state, game):
    """
    Take action which brings us closest to a candy
    Checks if we're hitting another snake
    """
    head = state.snakes[id].position[0]
    # Computing the list of actions that won't kill the snake
    actions = [m for m in game.actions(state, id)
               if add(head, m) not in
               sum([state.snakes[other].position for other in state.snakes.keys() if other != id], [])]

    # If it is empty, then the snake will die and we move randomly
    if len(actions) == 0:
        return random.sample(game.actions(state, id), 1)[0]

    # If there is no candy we move randomly
    if len(state.candies) == 0:
        return random.sample(actions, 1)[0]
    best_move = min((dist(add(head, move), candy), move) for candy in state.candies.keys() for move in actions)
    return best_move[1]