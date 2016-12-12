from strategies import randomStrategy, greedyStrategy, smartGreedyStrategy, opportunistStrategy
from rl import simpleFeatureExtractor1, simpleFeatureExtractor2, projectedDistances, projectedDistances2
from minimax import AlphaBetaAgent, ExpectimaxAgent, greedyEvaluationFunction, smartCowardDfunc, survivorDfunc

agent             = "RL"
filename          = "rl3-g20-pd2-1a"
discount          = 0.9
depth             = lambda s,a : survivorDfunc(s, a , 2, 0.5)
evalFn            = greedyEvaluationFunction
grid_size         = 20
max_iter          = 3000
num_trials        = 6000
opponents         = [smartGreedyStrategy, AlphaBetaAgent(evalFn, depth).getAction]
featureExtractor  = projectedDistances2
lambda_           = None