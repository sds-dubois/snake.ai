"""
Interface for the multi player snake game
"""

# imports
import utils
import random
import math

# global variables
MOVES = [(1,0), (0,1), (-1,0), (0,-1)]      # authorized moves
CANDY_VAL = 1                               # default candy value
CANDY_BONUS = 5                             # candy value for dead snakes


class Snake:
    """
    Snake object.
    Position is a list of (x,y) tuples from head to tail
    """
    def __init__(self, position):
        self.position = position
        self.points = 0
        self.last_tail = None

    def move(self, direction):
        self.last_tail = self.position[-1]
        head = utils.add(self.position[0], direction)
        self.position = [head] + self.position[:-1]

    def size(self):
        return len(self.position)

    def orientation(self):
        return utils.add(self.position[0], self.position[1], mu = -1)

    def addPoints(self, val):
        self.points += val
        # check if size increases
        if val == CANDY_BONUS or self.points % CANDY_BONUS:
            self.position.append(self.last_tail)


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
    Defined by a dictionary {id => snake} and {position => value} for candies.
    """
    def __init__(self, snakes, candies):
        self.snakes = snakes
        self.candies = dict((c.position, c.value) for c in candies)
        self.iter = 0

    def addCandy(self, pos, val):
        """
        Adds a candy of value val and position pos. If there is already a snake at the position, we don't add it
        :param pos: the position for the candy as a tuple
        :param val: the value of the candy
        :return: True if the candy has been added, False if not
        """
        if not pos in [p for s in self.snakes.keys() for p in self.snakes[s].position] \
                and not pos in self.candies.keys():
            self.candies[pos] = val
            return True
        return False

    def addNRandomCandies(self, n, grid_size):
        while n > 0:
            if self.addCandy(
                    (random.randint(0, grid_size), random.randint(0, grid_size)),
                    CANDY_VAL
            ):
                n -= 1

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
                self.snakes[id].add_points(self.candies.get(head))
                del self.candies[head]

        # remove snakes which bumped into other snakes
        deads = []
        for id in self.snakes.keys():
            # list of (x,y) points occupied by other snakes
            otherSnakes = [p for s in self.snakes.keys() for p in self.snakes[s].position if s != id]
            if self.snakes[id].position[0] in otherSnakes:
                deads.append(id)
                # add candy at head's position before last move
                self.candies[self.snakes[id].position[1]] = CANDY_BONUS

        for id in deads:
            print "Snake {} died with {} points".format(id, self.snakes[id].points)
            del self.snakes[id]


class Game:
    def __init__(self, grid_size, n_snakes=2, candy_ratio=1., max_iter = None):
        self.grid_size = grid_size
        self.max_iter = max_iter
        self.n_snakes = n_snakes
        self.candy_ratio = candy_ratio

    def isEnd(self, state):
        if self.max_iter:
            return len(state.n_snakes) == 1 or state.iter == self.max_iter
        else:
            return len(state.snakes) == 1

    def startState(self):
        """
        Initialize a game with `n_snakes` snakes of size 2, randomly assigned to different locations of the grid,
        and `n_candies` candies, randomly located over the grid.
        Guarantees a valid state.
        """
        log_base2 = math.ceil(math.log(self.n_snakes)/math.log(2))
        n_squares = int(2 ** log_base2)
        square_size = self.grid_size / n_squares
        assignment = random.sample(range(n_squares), self.n_snakes)

        assert self.grid_size >= 3 * log_base2

        snakes = {}
        for snake, assign in enumerate(assignment):
            rand_pos = (random.randint(1, square_size-2),
                        random.randint(1, square_size-2))
            head = (rand_pos[0] + (n_squares / (assign+1))*square_size,
                    rand_pos[1] + (n_squares % (assign+1))*square_size)
            snakes[snake] = Snake([head, utils.add(head, random.sample(MOVES, 1)[0])])

        candies_to_put = 2*int(self.candy_ratio)+1
        start_state = State(snakes, {})
        start_state.addNRandomCandies(candies_to_put, self.grid_size)
        return start_state


    def isOnGrid(self, p):
        """
        Check if position `p` is valid for the grid.
        """
        return p[0] > 0 and p[1] > 0 and p[0] < self.grid_size and p[1] < self.grid_size

    def actions(self, state, player):
        """
        List of possible actions for `player`.
        """
        snake = state.snakes.get(player)
        head = snake.position[0]
        return [m for m in MOVES if m != utils.mult(snake.orientation(), -1) and self.isOnGrid(utils.add(head, m))]

    def succ(self, state, actions):
        """
        `actions` is a dict {snake_id => move}
        Update snakes' position and randomly add some candies.
        """
        state = state.update(actions)
        rand_pos = (random.randint(1, self.grid_size[0]-1), random.randint(1, self.grid_size[1]-1))
        state.addCandy(rand_pos, CANDY_VAL)
        return state

