"""
Interface for the multiplayer snake game
"""

# imports
import utils

# global variables
MOVES = [(1,0), (0,1), (-1,0), (0,-1)]


class Snake:
    """
    Snake object.
    Position is a list of (x,y) tuples from head to tail
    """
    def __init__(self, position):
        self.position = position
        self.points = 0

    def move(self, direction):
        head = add(self.position[0], direction)
        self.position = [head] + self.position[:-1]

    def size(self):
        return len(self.position)

    def orientation(self):
        return add(self.position[0], self.position[1], lambda = -1)


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
    Defined by a dictionnary {id => snake} and {position => value} for candies.
    """
    def __init__(self, snakes, candies):
        self.snakes = snakes
        self.candies = dict((c.position, c.value) for c in candies)
        self.iter = 0

    def is_end(self, max_iter = None):
        if max_iter:
            return len(self.snakes) == 1 or self.iter == max_iter 
        else:
            return len(self.snakes) == 1

    def update(self, moves):
        """
        `moves` is a dict {snake_id => move}
        Update the positions/points of every snakes and check for collisions.
        """
        self.iter += 1

        # update positions
        for id, m in moves.iteritems():
            self.snakes[id].move(m)
            head = self.snakes[id].position[0]
            if head in self.candies:
                self.snakes[id].points += self.candies.get(head)
                del self.candies[head]

        # remove snakes which bumped into other snakes
        deads = []
        for id in self.snakes.keys():
            # list of (x,y) points occupied by other snakes
            otherSnakes = [p for s in self.snakes.keys() for p in self.snakes[s].position if s != id]
            if self.snakes[id].position[0] in otherSnakes:
                deads.append(id)
                # add candy
                self.candies[self.snakes[id].position[0]] = 5

        for id in deads:
            print "Snake {} died with {} points".format(id, self.snakes[id].points)
            del self.snakes


class Game:
    def __init__(self, grid_size):
        self.grid_size = grid_size
    
    def startState(self, n_snakes, n_candies):
        """
        Initialize a game with `n_snakes` snakes of size 2 
        and `n_candies` candies, randomly located over the grid.
        Guarantees a valid state.
        """
        # TODO: make it random and allow more snakes
        assert n_snakes == 2
        assert grid_size[0] > 10 and grid_size[1] > 10
        snakes = {"0": [(2,2), (2,1)], "1": [(8,8), (9,8)]}
        candies = [Candy((4,4), 1), Candy((3,4), 1), Candy((6,1), 1)]
        return State(snakes, candies)

    def isOnGrid(self, p):
        """
        Check if position `p` is valid for the grid.
        """
        return p[0] > 0 and p[1] > 0 and p[0] < self.grid_size[0] and p[1] < self.grid_size[1]

    def actions(self, state, player):
        """
        List of possible actions for `player`.
        """
        snake = state.snakes.get(player)
        head = snake.position[0]
        return [m for m in MOVES if m != mult(snake.orientation(), -1) and self.isOnGrid(add(head, m))]

    def succ(self, state, actions):
        """
        `actions` is a dict {snake_id => move}
        Update snakes' position and randomly add some candies.
        """
        # TODO: add candies
        return state.update(actions)

