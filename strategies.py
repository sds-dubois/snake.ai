"""
Strategies for players.
"""

from utils import *
import random

def randomStrategy(id, state, game):
    """
    Takes random actions
    """
    return random.sample(game.actions(state, id), 1)[0]


def greedyStrategy(id, state, game):
    """
    Take action which brings us closest to a candy - without even 
    looking at other snakes. 
    """
    actions = game.simple_actions(state, id)
    head = state.snakes[id].position[0]
    if len(state.candies) == 0:
        return random.sample(actions, 1)[0]
    best_move = min((dist(add(head, move, mu=norm), candy), (move, norm))
                    for candy in state.candies.keys() for move, norm in actions)
    return best_move[1]

def smartGreedyStrategy(id, state, game):
    """
    Take action which brings us closest to a candy
    Checks if we're hitting another snake
    """
    snake = state.snakes[id]
    head = snake.position[0]
    # Computing the list of actions that won't kill the snake
    actions = [move for move in game.simple_actions(state, id)
               if snake.predictHead(move) not in
               sum([state.snakes[other].position for other in state.snakes.keys() if other != id], [])]

    # If it is empty, then the snake will die and we move randomly
    if len(actions) == 0:
        return random.sample(game.simple_actions(state, id), 1)[0]

    # If there is no candy we move randomly
    if len(state.candies) == 0:
        return random.sample(actions, 1)[0]
    best_move = min((dist(snake.predictHead(move), candy), move)
                    for candy in state.candies.keys() for move in actions)
    return best_move[1]

def opportunistStrategy(id, state, game):
    """
    Take action which brings us closest to a candy
    Checks if we're hitting another snake
    """
    snake = state.snakes[id]
    head = snake.position[0]
    # Computing the list of actions that won't kill the snake
    actions = [m for m in game.simple_actions(state, id)
               if snake.predictHead(m) not in
               sum([state.snakes[other].position for other in state.snakes.keys() if other != id], [])]

    # If it is empty, then the snake will die and we move randomly
    if len(actions) == 0:
        return random.sample(game.simple_actions(state, id), 1)[0]

    # If there is no candy we move randomly
    if len(state.candies) == 0:
        return random.sample(actions, 1)[0]
    
    min_dist = dict((candy, min(dist(s.position[0], candy) for s in state.snakes.values())) for candy in state.candies.keys())
    best_move = min((dist(snake.predictHead(move), candy) - min_dist[candy],
                     dist(snake.predictHead(move), candy), move) for candy in state.candies.keys() for move in actions)
    return best_move[2]