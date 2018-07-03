from move import Move
from agent import Agent
from strategies import *

# global variables
ACCELERATION = False
DIRECTIONS = [(1,0), (0,1), (-1,0), (0,-1)]      # authorized moves
NORM_MOVES = [1]
if ACCELERATION:
    NORM_MOVES.append(2)                    # acceleration moves
MOVES = [Move(dir, norm) for dir in DIRECTIONS for norm in NORM_MOVES]
NO_MOVE = Move(direction = (0,0), norm = 0)
CANDY_VAL = 1                               # default candy value
CANDY_BONUS = 3                             # candy value for dead snakes

RandomAgent = Agent(name = "RandomAgent", strategy = randomStrategy)
GreedyAgent = Agent(name = "GreedyAgent", strategy = greedyStrategy)
SmartGreedyAgent = Agent(name = "SmartGreedyAgent", strategy = smartGreedyStrategy)
OpportunistAgent = Agent(name = "OpportunistAgent", strategy = opportunistStrategy)
HumanAgent = Agent(name = "human", strategy = humanStrategy)