"""
Strategies for players.
"""

def greedy(id, state, game):
    """
    Take action which brings us closest to a candy - without even 
    looking at other snakes. 
    """
    actions = game.actions(state, id)
    head = state.snakes[id].position[0]
    utilities = [min(dist(add(head, move), candy), move for candy in state.candies.keys()) for move in actions]
    return min(utilities)[1]