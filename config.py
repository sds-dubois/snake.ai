from strategies import randomStrategy, greedyStrategy, smartGreedyStrategy, opportunistStrategy
from rl import simpleFeatureExtractor1, simpleFeatureExtractor2, projectedDistances
from minimax import AlphaBetaAgent, ExpectimaxAgent, greedyEvaluationFunction

PARAMS = {
    "agent"            : "RL",
    "discount"         : 0.9,
    "opponents"        : [randomStrategy, smartGreedyStrategy, opportunistStrategy],
    "featureExtractor" : simpleFeatureExtractor1, # projectedDistances,
    "lambda_"          : 0.4,
    "depth"            : None,
    "evalFn"           : None,
    "grid_size"        : 20,
    "num_trials"       : 1000,
    "max_iter"         : 3000,
    "filename"         : "nr3-td-weights1c.p"
}