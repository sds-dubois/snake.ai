from strategies import randomStrategy, greedyStrategy, smartGreedyStrategy, opportunistStrategy
from minimax import AlphaBetaAgent, ExpectimaxAgent, greedyEvaluationFunction, smartCowardDfunc, survivorDfunc

agent             = "RL"
filename          = "nn2-adam-r15-l20-1a"
discount          = 0.9
depth             = lambda s,a : survivorDfunc(s, a , 2, 0.5)
evalFn            = greedyEvaluationFunction
grid_size         = 20
max_iter          = 3000
num_trials        = 8000
# opponents         = [smartGreedyStrategy, opportunistStrategy]
opponents         = [smartGreedyStrategy, opportunistStrategy, AlphaBetaAgent(evalFn, depth).getAction]
radius            = 15
lambda_           = None
rl_type           = "nn"
comment           = "d2"