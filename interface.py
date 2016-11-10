"""
Interface for the multi player snake game
"""

# imports
import utils
import random
import math

# global variables
ACCELERATION = True
MOVES = [(1,0), (0,1), (-1,0), (0,-1)]      # authorized moves
NORM_MOVES = [1]
if ACCELERATION:
    NORM_MOVES.append(2)                    # acceleration moves
CANDY_VAL = 1                               # default candy value
CANDY_BONUS = 2                             # candy value for dead snakes


class Snake:
    """
    Snake object.
    Position is a list of (x,y) tuples from head to tail
    """
    def __init__(self, position):
        self.position = position
        self.points = 2*CANDY_BONUS
        self.size = 2
        self.last_tail = None

    def predictHead(self, move):
        direction, norm = move
        return utils.add(self.position[0], direction, mu=norm)

    def move(self, direction, norm):
        '''
        Moves according the direction vectors, if it accelerates, returns the position to put a candy on
        :param direction: the tuple encoding the direction
        :param norm: if 1 normal move, if 2 acceleration
        :return: None if the snake didn't accelerate, the position to put a candy on, if it did accelerate
        '''
        if norm == 2:
            self.last_tail = self.position[-2]
            second = utils.add(self.position[0], direction)
            head = utils.add(second, direction)
            self.position = [head, second] + self.position[:-2]
            self.removePoints(CANDY_VAL)
            return self.last_tail

        self.last_tail = self.position[-1]
        head = utils.add(self.position[0], direction)
        self.position = [head] + self.position[:-1]
        return None


    def size(self):
        return len(self.position)

    def orientation(self):
        return utils.add(self.position[0], self.position[1], mu = -1)

    def addPoints(self, val):
        self.points += val
        # check if size increases
        if self.points / CANDY_BONUS > self.size:
            self.position.append(self.last_tail)
            self.size += 1

    def removePoints(self, val):
        self.points -= val
        # check if size decreases
        if self.points / CANDY_BONUS < self.size:
            self.last_tail = self.position[-1]
            del self.position[-1]
            self.size -= 1



class State:
    """
    State object for the multiplayer snake game.
    Defined by a dictionary {id => snake} and {position => value} for candies.
    """

    grid_size = None

    def __init__(self, snakes, candies):
        self.snakes = snakes
        self.candies = dict((c.position, c.value) for c in candies)
        self.iter = 0

    def __str__(self):
        s = "--- state {} ---\n".format(self.iter)
        s += "- snakes:\n"
        s += "\n".join(["\t{}:\t{}\t-\t{}".format(id, s.points, s.position) for id,s in self.snakes.iteritems()])
        s += "\n- candies:\n"
        s += "\n".join(["\t{}\t{}".format(v, pos) for pos,v in self.candies.iteritems()])
        return s

    def shape(self, i, j):
        if (i,j) in self.candies:
            if self.candies[(i,j)] == CANDY_BONUS:
                return '+'
            return '*'
        for id, s in self.snakes.iteritems():
            if (i,j) == s.position[0]:
                return '@'
            if (i,j) in s.position[1:]:
                return str(id)
        return ' '

    def printGrid(self, grid_size):
        s = "--- state {} ---\n".format(self.iter)
        s += "-" * (grid_size + 1) + '\n'
        for i in range(grid_size):
            s += '|' + ''.join(self.shape(i,j) for j in range(grid_size)) + '|\n'
        s += "-" * (grid_size + 1)+ '\n'
        print s

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
        candies_to_add = []
        accelerated = {}
        for id, (dir, norm) in moves.iteritems():
            new_candy_pos = self.snakes[id].move(dir, norm)

            # We remember where to add candies when the snake accelerated
            if new_candy_pos is not None:
               candies_to_add.append(new_candy_pos)

            # We collect candies if head touches a candy
            head = self.snakes[id].position[0]
            if head in self.candies:
                self.snakes[id].addPoints(self.candies.get(head))
                del self.candies[head]

            # If the snake accelerated, we check if the second part of the body touches a candy
            if norm == 2:
                accelerated[id] = True
                second = self.snakes[id].position[1]
                if second in self.candies:
                    self.snakes[id].addPoints(self.candies.get(second))
                    del self.candies[second]
            else:
                accelerated[id] = False

        # add candies created by acceleration
        for cand_pos in candies_to_add:
            self.addCandy(cand_pos, CANDY_VAL)

        # remove snakes which bumped into other snakes
        deads = []
        for id in self.snakes.keys():
            # list of (x,y) points occupied by other snakes
            otherSnakes = [p for s in self.snakes.keys() for p in self.snakes[s].position if s != id]
            if self.snakes[id].position[0] in otherSnakes\
                    or (accelerated[id] and self.snakes[id].position[1] in otherSnakes)\
                    or not utils.isOnGrid(self.snakes[id].position[0], self.grid_size):
                deads.append(id)
                # add candies on the snake position before last move
                for p in self.snakes[id].position:
                    self.candies[p] = CANDY_BONUS

        for id in deads:
            # print "Snake {} died with {} points".format(id, self.snakes[id].points)
            del self.snakes[id]

        return self


class Game:
    def __init__(self, grid_size, n_snakes = 2, candy_ratio = 1., max_iter = None):
        self.grid_size = grid_size
        self.max_iter = max_iter
        self.n_snakes = n_snakes
        self.candy_ratio = candy_ratio
        State.grid_size = grid_size

    def startState(self):
        """
        Initialize a game with `n_snakes` snakes of size 2, randomly assigned to different locations of the grid,
        and `n_candies` candies, randomly located over the grid.
        Guarantees a valid state.
        """
        n_squares_per_row = int(math.ceil(math.sqrt(self.n_snakes))**2)
        square_size = self.grid_size / int(n_squares_per_row)
        assignment = random.sample(range(n_squares_per_row ** 2), self.n_snakes)

        assert self.grid_size >= 3*n_squares_per_row

        snakes = {}
        for snake, assign in enumerate(assignment):
            head = (random.randint(1, square_size-2) + (assign / n_squares_per_row) * square_size,
                    random.randint(1, square_size-2) + (assign % n_squares_per_row) * square_size)
            snakes[snake] = Snake([head, utils.add(head, random.sample(MOVES, 1)[0])])

        candies_to_put = 2 * int(self.candy_ratio) + 1
        start_state = State(snakes, {})
        start_state.addNRandomCandies(candies_to_put, self.grid_size)
        return start_state

    def isEnd(self, state):
        if self.max_iter:
            return len(state.snakes) <= 1 or state.iter == self.max_iter
        else:
            return len(state.snakes) <= 1

    def actions(self, state, player):
        """
        List of possible actions for `player`.
        """
        snake = state.snakes.get(player)
        head = snake.position[0]
        return [(m,n) for m in MOVES for n in NORM_MOVES
                if m != utils.mult(snake.orientation(), -1)
                and (n == 1 or snake.size > 2)
                and utils.isOnGrid(utils.add(head, m, mu=n), self.grid_size)]

    def simple_actions(self, state, player):
        """
        List of possible actions for `player`.
        """
        snake = state.snakes.get(player)
        head = snake.position[0]
        return [(m,1) for m in MOVES if m != utils.mult(snake.orientation(), -1)
                and utils.isOnGrid(utils.add(head, m), self.grid_size)]

    def succ(self, state, actions):
        """
        `actions` is a dict {snake_id => move}
        Update snakes' position and randomly add some candies.
        """
        state = state.update(actions)
        rand_pos = (random.randint(1, self.grid_size-1), random.randint(1, self.grid_size-1))
        state.addCandy(rand_pos, CANDY_VAL)
        return state

