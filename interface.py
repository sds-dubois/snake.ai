"""
Interface for the multiplayer snake game
"""

class Snake:
    """
    Snake object.
    Position is a list of (x,y) tuples from head to tail
    """
    def __init__(self, position):
        self.position = position
    
    def move(self, direction):
        # TODO: update position
        return None

    def size(self):
        return len(self.position)

    def orientation(self):
        return (self.position[1][0] - self.position[1][0], self.position[0][1] - self.position[1][1])


class Candy:
    """
    Candy object.
    Defined by its (x,y) position and value.
    """
    def __init__(self, position, value):
        self.position = position
        self.value = value


class State:
    """
    State object for the multiplayer snake game.
    Defined by a dictionnary {id => snake} and list of candies.
    """
    def __init__(self, snakes, candies):
        self.snakes = snakes
        self.candies = candies

    def is_end(self):
        return len(self.snakes) == 1

    def update(self, moves):
        """
        moves is a dict {snake_id => move}
        Update the positions of every snakes and check for collisions.
        """
        # TODO
        return None


class Game:
    def __init__(self, grid_size):
        self.grid_size = grid_size
    
    def startState(self, n_snakes, n_candies):
        """
        Initialize a game with `n_snakes` snakes and `n_candies` candies,
        randomly located over the grid.
        Guarantees a valid state.
        """
        # TODO
        return None
    
    def actions(self, state, player):
        """
        List of possible actions for `player`.
        """
        # TODO
        return None

    def succAndProb(self, state, actions):
        """
        `actions` is a dict {snake_id => move}
        Update snakes' position and randomly add some candies.
        """
        # TODO
        return None
    
