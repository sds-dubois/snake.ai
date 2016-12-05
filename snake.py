__author__ = 'The Real Seb'

import numpy as np
from collections import deque
import utils
from constants import NORM_MOVES, CANDY_VAL, CANDY_BONUS

class newSnake:
    grid_size = None

    def __init__(self, position, i=0):
        self.position = deque(position)
        self.points = len(position)*CANDY_BONUS
        self.on_tail = False
        self.last_tail = None
        self.bool_pos = np.zeros((self.grid_size, self.grid_size))
        self.id = i
        for pos in position:
            self.bool_pos[pos] = 1

    def head(self):
        return self.position[0]

    def predictHead(self, move):
        return move.apply(self.head())

    def onSnake(self, pos):
        return self.bool_pos[pos] > 0

    def onSnakeOrNotGrid(self, pos):
        return not utils.isOnGrid(pos, self.grid_size) or self.onSnake(pos)

    def countSnake(self, pos):
        return self.bool_pos[pos]

    def onSnakeExceptLastOrNotGrid(self, pos, n):
        return not utils.isOnGrid(pos, self.grid_size) or \
               (self.countSnake(pos) - sum(int(self.position[-i] == pos) for i in xrange(1,n+1)) >= 1)

    def pop(self):
        tail = self.position.pop()
        self.bool_pos[tail] -= 1
        self.last_tail = tail
        return tail

    def popleft(self):
        head = self.position.popleft()
        self.bool_pos[head] -= 1

    def add(self, pos):
        self.bool_pos[pos] += 1
        self.position.appendleft(pos)

    def addRight(self, pos):
        self.bool_pos[pos] += 1
        self.position.append(pos)

    def isInArea(self, pos, radius):
        for i in xrange(max(-radius+pos[0],0), min(radius+pos[0]+1, self.grid_size)):
            for j in xrange(max(-radius+pos[1],0), min(radius+pos[1]+1, self.grid_size)):
                if self.onSnake((i,j)):
                    return True
        return False

    def authorizedMove(self, move, possibleNorm=NORM_MOVES):
        '''
        Returns if the move is authorized given a optional direction for the collision constraints
        :param move: the move to check
        :param possibleNorm: check only the norm provided
        :return: a boolean true if the position is authorized
        '''
        head = self.head()

        # backward moves are forbidden
        if move.direction() == utils.mult(self.orientation(), -1):
            return False

        target = move.applyDirection(head)
        # If a collision already occurred we can't do another one
        if (self.on_tail and self.onSnakeExceptLastOrNotGrid(target, 1)):
            return False
        # If we would need two collisions in a row there is a problem
        next_target = move.applyDirection(head, mu=2)
        if (self.onSnakeExceptLastOrNotGrid(target, 1) and
            self.onSnakeExceptLastOrNotGrid(next_target, 2)):
            return False

        if move.norm() == 2 and 2 in possibleNorm:
            # We can only accelerate when the snake is big enough
            if self.size() <= 2:
                return False

            next_acc_target = move.applyDirection(head, mu=3)
            # We make sure that we can move without causing death at the next time
            if (self.onSnakeExceptLastOrNotGrid(next_acc_target, 3) and
                self.onSnakeExceptLastOrNotGrid(next_target)):
                return False

        return True

    def backward(self, last_pos, last_tail):
        for pos in last_pos:
            self.popleft()
            self.addRight(pos)
        if len(last_pos) == 2:
            self.addPoints(CANDY_VAL)
        self.last_tail = last_tail
        self.on_tail = (self.countSnake(self.head()) >= 2)

    def move(self, move):
        '''
        Moves according the direction vectors, if it accelerates, returns the position to put a candy on
        :param move: a (direction, norm) tuple with direction being the tuple encoding the direction
        and norm being 1 for a normal move and 2 for acceleration
        :return: None if the snake didn't accelerate, the position to put a candy on, if it did accelerate
        '''
        norm, direction = move.norm(), move.direction()
        self.on_tail = False
        if norm == 2:
            self.pop()
            before_last_tail = self.pop()
            self.add(utils.add(self.position[0], direction))
            new_head = utils.add(self.position[0], direction)
            if self.onSnake(new_head):
                self.on_tail = True
            self.add(new_head)
            self.removePoints(CANDY_VAL)
            return before_last_tail

        self.pop()
        head = utils.add(self.head(), direction)
        if not utils.isOnGrid(head, self.grid_size):
            print head
            print self.id
            print self.position
            print self.bool_pos
            print self
        if self.onSnake(head):
            self.on_tail = True
        self.add(head)
        return None

    def __len__(self):
        return len(self.position)

    def size(self):
        return len(self)

    def orientation(self):
        return utils.add(self.position[0], self.position[1], mu = -1)

    def addPoints(self, val):
        self.points += val
        # check if size increases
        if self.points / CANDY_BONUS > self.size():
            self.addRight(self.last_tail)

    def removePoints(self, val):
        self.points -= val
        # check if size decreases
        if self.points / CANDY_BONUS < self.size():
            tail = self.pop()
            self.last_tail = tail

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
        self.on_tail = False

    def head(self):
        return self.position[0]

    def predictHead(self, move):
        return move.apply(self.position[0])

    def authorizedMove(self, move, possibleNorm=NORM_MOVES):
        '''
        Returns if the move is authorized given a optional direction for the collision constraints
        :param move: the move to check
        :param possibleNorm: check only the norm provided
        :return: a boolean true if the position is authorized
        '''
        head = self.position[0]

        # backward moves are forbidden
        if move.direction() == utils.mult(self.orientation(), -1):
            return False

        # If a collision already occurred we can't do another one
        if (self.on_tail and move.applyDirection(head) in self.position[:-1]):
            return False
        # If we would need two collisions in a row there is a problem
        if (move.applyDirection(head) in self.position[:-1] and move.applyDirection(head, mu=2) in self.position[:-2]):
            return False

        if move.norm() == 2 and 2 in possibleNorm:
            # We can only accelerate when the snake is big enough
            if self.size <= 2:
                return False

            # We make sure that we can move without causing death at the next time
            if (move.applyDirection(head, mu=3) in self.position[:-3]
                and move.applyDirection(head, mu=2) in self.position[:-2]):
                return False

        return True

    def move(self, move):
        '''
        Moves according the direction vectors, if it accelerates, returns the position to put a candy on
        :param move: a (direction, norm) tuple with direction being the tuple encoding the direction
        and norm being 1 for a normal move and 2 for acceleration
        :return: None if the snake didn't accelerate, the position to put a candy on, if it did accelerate
        '''
        norm, direction = move.norm(), move.direction()
        self.on_tail = False
        if norm == 2:
            self.last_tail = self.position[-2]
            second = utils.add(self.position[0], direction)
            head = utils.add(second, direction)
            self.position = [head, second] + self.position[:-2]
            self.removePoints(CANDY_VAL)
            if head in self.position[1:]:
                self.on_tail = True
            return self.last_tail

        self.last_tail = self.position[-1]
        head = utils.add(self.position[0], direction)
        self.position = [head] + self.position[:-1]
        if head in self.position[1:]:
                self.on_tail = True
        return None

    def onSnake(self, pos):
        return pos in self.position

    def countSnake(self, pos):
        return self.position.count(pos)

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
