from strategies import randomStrategy, greedyStrategy, smartGreedyStrategy, opportunistStrategy
from rl import simpleFeatureExtractor1, simpleFeatureExtractor2, projectedDistances
from minimax import AlphaBetaAgent, ExpectimaxAgent, greedyEvaluationFunction

PARAMS = {
    "agent"            : "RL",
    "opponents"        : [randomStrategy, smartGreedyStrategy, opportunistStrategy],
    "featureExtractor" : projectedDistances,
    "lambda_"          : 0.2,
    "depth"            : None,
    "evalFn"           : None,
    "grid_size"        : 20,
    "num_trials"       : 5000,
    "max_iter"         : 3000,
    "filename"         : "td-weights5.p"
}