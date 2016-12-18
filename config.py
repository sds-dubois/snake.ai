from strategies import randomStrategy, greedyStrategy, smartGreedyStrategy, opportunistStrategy
from rl import simpleFeatureExtractor1, simpleFeatureExtractor2, projectedDistances, projectedDistances2, projectedDistances3
from minimax import AlphaBetaAgent, ExpectimaxAgent, greedyEvaluationFunction, smartCowardDfunc, survivorDfunc

agent             = "RL"
filename          = "rl4-g20-pd3-2b"
discount          = 0.9
depth             = lambda s,a : survivorDfunc(s, a , 2, 0.5)
evalFn            = greedyEvaluationFunction
grid_size         = 20
max_iter          = 3000
num_trials        = 10000
opponents         = [smartGreedyStrategy, opportunistStrategy]
featureExtractor  = projectedDistances3
lambda_           = None