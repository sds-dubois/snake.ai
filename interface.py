"""
Interface for the multi player snake game
"""

# imports
import random, math, copy
import utils
import numpy as np
from collections import deque
from time import time
from copy import deepcopy
from move import Move
from snake import Snake, newSnake
from constants import ACCELERATION, DIRECTIONS, NORM_MOVES, MOVES, CANDY_VAL, CANDY_BONUS


class State:
    """
    State object for the multiplayer snake game.
    Defined by a dictionary {id => snake} and {position => value} for candies.
    """

    grid_size = None
    n_snakes = 0
    max_iter = None
    time_copying = 0.0

    def __init__(self, snakes, candies):
        self.snakes = snakes
        self.candies = dict((c.position, c.value) for c in candies)
        self.scores = {}
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
                return ' +'
            return ' *'
        for id, s in self.snakes.iteritems():
            if (i,j) == s.position[0]:
                return ' @'
            c = s.countSnake((i,j))
            if c == 1:
                return ' {}'.format(id)
            if c == 2:
                return " #"
        return '  '

    def printGrid(self, grid_size = None):
        if grid_size is None:
            grid_size = self.grid_size
        s = "--- state {} ---\n".format(self.iter)
        s += "-" * 2*(grid_size + 1) + '\n'
        for i in range(grid_size):
            s += '|' + ''.join(self.shape(i,j) for j in range(grid_size)) + '|\n'
        s += "-" * 2*(grid_size + 1)+ '\n'
        print s


    def addCandy(self, pos, val, dead_snake=-1):
        """
        Adds a candy of value val and position pos. If there is already a snake at the position, we don't add it
        :param pos: the position for the candy as a tuple
        :param val: the value of the candy
        :return: True if the candy has been added, False if not
        """
        if all(not s.onSnake(pos) for a, s in self.snakes.iteritems() if a != dead_snake) \
                and not pos in self.candies.keys():
            self.candies[pos] = val
            return True
        return False

    def addNRandomCandies(self, n, grid_size):
        while n > 0:
            if self.addCandy(
                    (random.randint(0, grid_size-1), random.randint(0, grid_size-1)),
                    CANDY_VAL
            ):
                n -= 1

    def onOtherSnakes(self, pos, id):
        return any(s.onSnake(pos) for i,s in self.snakes.iteritems() if i != id)

    def oneAgentUpdate(self, id, m):
        #Remember changes
        snake_who_died = None
        candies_to_add = []
        candies_removed = []
        points_won = 0
        last_tail = self.snakes[id].last_tail
        last_pos = []

        # update positions

        accelerated = {}
        # If the snake couldn't move, then it's dead
        if m is None:
            snake_who_died = deepcopy(self.snakes[id])
        else:
            if m.norm() == 2:
                last_pos.append(self.snakes[id].position[-2])
            last_pos.append(self.snakes[id].position[-1])
            new_candy_pos = self.snakes[id].move(m)

            # We remember where to add candies when the snake accelerated
            if new_candy_pos is not None:
               candies_to_add.append(new_candy_pos)

            # We collect candies if head touches a candy
            head = self.snakes[id].head()
            if head in self.candies:
                points_won += self.candies.get(head)
                candies_removed.append((head, self.candies.get(head)))
                self.snakes[id].addPoints(self.candies.get(head))
                del self.candies[head]

            # If the snake accelerated, we check if the second part of the body touches a candy
            if m.norm() == 2:
                accelerated[id] = True
                second = self.snakes[id].position[1]
                if second in self.candies:
                    points_won += self.candies.get(second)
                    candies_removed.append((second, self.candies.get(second)))
                    self.snakes[id].addPoints(self.candies.get(second))
                    del self.candies[second]
            else:
                accelerated[id] = False

        # add candies created by acceleration
        for cand_pos in candies_to_add:
            self.addCandy(cand_pos, CANDY_VAL)

        # remove snakes which bumped into other snakes
        # list of (x,y) points occupied by other snakes
        if snake_who_died is None and (self.onOtherSnakes(self.snakes[id].position[0], id)\
                or (accelerated[id] and self.onOtherSnakes(self.snakes[id].position[1], id))\
                or not utils.isOnGrid(self.snakes[id].position[0], self.grid_size)):
            snake_who_died = deepcopy(self.snakes[id])


        if snake_who_died is not None:
            # add candies on the snake position before last move
            self.snakes[id].popleft()
            for p in self.snakes[id].position:
                if self.addCandy(p, CANDY_BONUS, dead_snake=id):
                    candies_to_add.append(p)
            # print "Snake {} died with {} points".format(id, self.snakes[id].points)
            del self.snakes[id]

        return last_pos, id, candies_to_add, candies_removed, points_won, last_tail, snake_who_died

    def reverseChanges(self, changes):
        last_pos, id, candies_added, candies_removed, points_won, last_tail, snake_who_died = changes
        if snake_who_died is not None:
            self.snakes[id] = snake_who_died
        self.snakes[id].removePoints(points_won)
        self.snakes[id].backward(last_pos, last_tail)
        for c in set(candies_added):
            del self.candies[c]
        for c, val in candies_removed:
            self.addCandy(c, val)


    def update(self, moves):
        """
        `moves` is a dict {snake_id => move}
        Update the positions/points of every snakes and check for collisions.
        """
        self.iter += 1

        deads = []

        # update positions
        candies_to_add = []
        accelerated = {}
        for id, m in moves.iteritems():
            # If the snake couldn't move, then it's dead
            if m is None or not self.snakes[id].authorizedMove(m):
                deads.append(id)
                continue

            new_candy_pos = self.snakes[id].move(m)

            # We remember where to add candies when the snake accelerated
            if new_candy_pos is not None:
               candies_to_add.append(new_candy_pos)

            # We collect candies if head touches a candy
            head = self.snakes[id].head()
            if head in self.candies:
                self.snakes[id].addPoints(self.candies.get(head))
                del self.candies[head]

            # If the snake accelerated, we check if the second part of the body touches a candy
            if m.norm() == 2:
                accelerated[id] = True
                second = self.snakes[id].position[1]
                if second in self.candies:
                    self.snakes[id].addPoints(self.candies.get(second))
                    del self.candies[second]
            else:
                accelerated[id] = False

        # add candies created by acceleration
        for cand_pos in candies_to_add:
            self.addCandy(cand_pos, CANDY_BONUS)

        # remove snakes which bumped into other snakes

        for id in moves.keys():
            # list of (x,y) points occupied by other snakes
            if not id in deads and (self.onOtherSnakes(self.snakes[id].position[0], id)\
                    or (accelerated[id] and self.onOtherSnakes(self.snakes[id].position[1], id))\
                    or not utils.isOnGrid(self.snakes[id].position[0], self.grid_size)):
                deads.append(id)

        # save scores and add candies
        rank = len(self.snakes)
        for id in deads:
            self.scores[id] = (rank, self.snakes[id].points) 
            # add candies on the snake position before last move
            for p in self.snakes[id].position:
                self.addCandy(p, CANDY_BONUS, dead_snake=id)
            # print "Snake {} died with {} points".format(id, self.snakes[id].points)
            del self.snakes[id]
        
        if len(self.snakes) == 1:
            winner = self.snakes.keys()[0]
            self.scores[winner] = (1, self.snakes[winner].points)

        return self

    def isWin(self, agent):
        return len(self.snakes) == 1 and agent in self.snakes.iterkeys()

    def isLose(self, agent):
        return len(self.snakes) >= 1 and agent not in self.snakes.iterkeys()

    def isDraw(self):
        return len(self.snakes) == 0

    def timesUp(self):
        return self.iter == self.max_iter

    def getNextAgent(self, agent, agents=None):
        if agents is None:
            agents = self.snakes.keys()
        else:
            agents = set(agents).intersection(set(self.snakes.iterkeys()))
        for i in range(1,self.n_snakes+1):
            next_snake = (agent+i) % self.n_snakes
            if next_snake in agents:
                return next_snake
        return agent

    def generateSuccessor(self, agent, move):
        return self.oneAgentUpdate(agent, move)

    def getScore(self, agent):
        if self.isDraw():
            return -1*(self.grid_size ** 2)*CANDY_BONUS+1
        if self.isWin(agent):
            return (self.grid_size ** 2)*CANDY_BONUS
        if self.timesUp():
            return self.snakes[agent].points
        if self.isLose(agent) or len(self.actions(agent)) == 0:
            return -1*(self.grid_size ** 2)*CANDY_BONUS
        return self.snakes[agent].points

    def currentScore(self, player):
        """
        Get the adjusted score for `player`: points/rank
        """
        s = self.scores.get(player)
        if s is None:
            return self.snakes[player].points / float(len(self.snakes))
        else:
            rank, points = s
            return points / float(rank)

    def actions(self, player):
        """
        List of possible actions for `player`.
        """
        snake = self.snakes.get(player)
        head = snake.position[0]
        return [m for m in MOVES
                if utils.isOnGrid(m.apply(head), self.grid_size)
                and snake.authorizedMove(m)]

    def simple_actions(self, player):
        """
        List of possible actions for `player`.
        """
        snake = self.snakes.get(player)
        head = snake.position[0]
        return [m for m in MOVES if m.norm() == 1
                and utils.isOnGrid(m.apply(head), self.grid_size)
                and snake.authorizedMove(m, possibleNorm=[1])]


class Game:
    def __init__(self, grid_size, n_snakes = 2, candy_ratio = 1., max_iter = None):
        self.grid_size = grid_size
        self.max_iter = max_iter
        self.n_snakes = n_snakes
        self.candy_ratio = candy_ratio


        # Update static variables of State
        State.grid_size = grid_size
        newSnake.grid_size = grid_size
        State.n_snakes = n_snakes
        State.max_iter = max_iter

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
            snakes[snake] = newSnake([head, utils.add(head, random.sample(DIRECTIONS, 1)[0])], snake)

        candies_to_put = 2 * int(self.candy_ratio) + 1
        start_state = State(snakes, {})
        start_state.addNRandomCandies(candies_to_put, self.grid_size)
        return start_state

    def isEnd(self, state):
        if self.max_iter:
            return len(state.snakes) <= 1 or state.iter == self.max_iter
        else:
            return len(state.snakes) <= 1


    def succ(self, state, actions, copy = True):
        """
        `actions` is a dict {snake_id => move}
        Update snakes' position and randomly add some candies.
        """
        if copy:
            newState = deepcopy(state)
        else:
            newState = state
        newState.update(actions)
        rand_pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
        newState.addCandy(rand_pos, CANDY_VAL)
        return newState

