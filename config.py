from strategies import randomStrategy, greedyStrategy, smartGreedyStrategy, opportunistStrategy
from minimax import AlphaBetaAgent, ExpectimaxAgent, greedyEvaluationFunction, smartCowardDfunc, survivorDfunc

agent             = "RL"
filename          = "nn1-r10-1a"
discount          = 0.9
depth             = lambda s,a : survivorDfunc(s, a , 2, 0.5)
evalFn            = greedyEvaluationFunction
grid_size         = 20
max_iter          = 3000
num_trials        = 3000
# opponents         = [smartGreedyStrategy, opportunistStrategy]
opponents         = [smartGreedyStrategy, opportunistStrategy, AlphaBetaAgent(evalFn, depth).getAction]
radius            = 10
lambda_           = None
rl_type           = "nn"