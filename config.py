from strategies import randomStrategy, greedyStrategy, smartGreedyStrategy, opportunistStrategy
from minimax import AlphaBetaAgent, ExpectimaxAgent, greedyEvaluationFunction, smartCowardDfunc, survivorDfunc

agent             = "ES"
filename          = "es-brs-r6-g20"
discount          = 0.99
depth             = lambda s,a : survivorDfunc(s, a , 2, 0.5)
evalFn            = greedyEvaluationFunction
grid_size         = 20
max_iter          = 400
num_trials        = 100
opponents         = [randomStrategy, opportunistStrategy]
# opponents         = [smartGreedyStrategy, opportunistStrategy, AlphaBetaAgent(evalFn, depth).getAction]
radius            = 6
lambda_           = None
rl_type           = "nn"
comment           = ""