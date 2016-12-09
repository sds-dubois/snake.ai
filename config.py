from strategies import randomStrategy, greedyStrategy, smartGreedyStrategy, opportunistStrategy
from rl import simpleFeatureExtractor1, simpleFeatureExtractor2, projectedDistances
from minimax import AlphaBetaAgent, ExpectimaxAgent, greedyEvaluationFunction, smartCowardDfunc, survivorDfunc

PARAMS = {
    "agent"            : "AlphaBeta",
    "discount"         : 0.9,
    "opponents"        : [smartGreedyStrategy, opportunistStrategy],
    "featureExtractor" : simpleFeatureExtractor1, # projectedDistances,
    "lambda_"          : 0.4,
    "depth"            : lambda s,a: survivorDfunc(s, a ,2, 0.6),
    "evalFn"           : greedyEvaluationFunction,
    "grid_size"        : 20,
    "num_trials"       : 1000,
    "max_iter"         : 3000,
    "filename"         : "ab-survivor2-3-5-0.6"
}