from hp import *
from strategies import randomStrategy, greedyStrategy, smartGreedyStrategy, opportunistStrategy
from minimax import AlphaBetaAgent, ExpectimaxAgent, greedyEvaluationFunction, smartCowardDfunc, survivorDfunc

agent             = "PG"
filename          = "pg-linear-r6-5000"
game_hp           = HP(grid_size = 20, max_iter = 3000, discount = 0.9)
rl_hp             = RlHp(radius = 6, filter_actions = True, lambda_ = None, q_type = "nn")
depth             = lambda s,a : survivorDfunc(s, a , 2, 0.5)
evalFn            = greedyEvaluationFunction
num_trials        = 5000
opponents         = [smartGreedyStrategy, opportunistStrategy]
# opponents         = [smartGreedyStrategy, opportunistStrategy, AlphaBetaAgent(evalFn, depth).getAction]
comment           = ""