from move import Move

# global variables
ACCELERATION = False
DIRECTIONS = [(1,0), (0,1), (-1,0), (0,-1)]      # authorized moves
NORM_MOVES = [1]
if ACCELERATION:
    NORM_MOVES.append(2)                    # acceleration moves
MOVES = [Move(dir, norm) for dir in DIRECTIONS for norm in NORM_MOVES]
CANDY_VAL = 1                               # default candy value
CANDY_BONUS = 3                             # candy value for dead snakes
