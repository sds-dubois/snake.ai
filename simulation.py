import sys
from time import sleep, time
import numpy as np
from controller import controller
from strategies import randomStrategy, greedyStrategy, smartGreedyStrategy, opportunistStrategy
from rl import rl_strategy, load_rl_strategy, simpleFeatureExtractor0, simpleFeatureExtractor1, simpleFeatureExtractor2
from utils import progressBar
from minimax import AlphaBetaAgent, ExpectimaxAgent, greedyEvaluationFunction
from interface import State


def simulate(n_simul, strategies, grid_size, candy_ratio = 1., max_iter = 500):
    print "Simulations"
    wins = dict((id, 0.) for id in xrange(len(strategies)))
    points = dict((id, []) for id in xrange(len(strategies)))
    iterations = []
    for it in xrange(n_simul):
        progressBar(it, n_simul)
        endState = controller(strategies, grid_size, candy_ratio = candy_ratio, max_iter = max_iter, verbose = 0)
        if len(endState.snakes) == 1:
            wins[endState.snakes.keys()[0]] += 1. / n_simul
            points[endState.snakes.keys()[0]].append(endState.snakes.values()[0].points)
        iterations.append(endState.iter)
    progressBar(n_simul, n_simul)
    points = dict((id, sum(val)/len(val)) for id,val in points.iteritems())
    return wins, points, iterations


if __name__ ==  "__main__":
    MAX_ITER = 1000

    if len(sys.argv) > 1:
        n_simul = int(sys.argv[1])
    else:
        n_simul = 1000

    alphabeta_agent = AlphaBetaAgent(depth=lambda s, a: 1, evalFn=greedyEvaluationFunction)
    expectimax_agent = ExpectimaxAgent(depth=lambda s, a: 1, evalFn=greedyEvaluationFunction)
    #rlStrategy = rl_strategy([randomStrategy, smartGreedyStrategy, opportunistStrategy], simpleFeatureExtractor1, 20, num_trials=10000, max_iter=3000, filename = "d-weights5.p")
    # rlStrategy = load_rl_strategy("weights3.p", [randomStrategy, smartGreedyStrategy, opportunistStrategy], simpleFeatureExtractor1)
    # strategies = [randomStrategy, greedyStrategy, smartGreedyStrategy, opportunistStrategy]
    strategies = [expectimax_agent.getAction, alphabeta_agent.getAction]
    t0 = time()
    wins, points, iterations = simulate(n_simul, strategies, 20, max_iter = MAX_ITER)
    print "Time spent: {}s".format(time()-t0)
    print "Time copying: {}s".format(State.time_copying)


    print "\n\n=======Results======="
    print "Run {} simulations".format(n_simul)
    print "Max iteration:", MAX_ITER, "\n"
    for i in range(len(strategies)):
        print "\t Snake {} wins {:.2f}% of the games, with {:.2f} points on average".format(i, wins[i]*100, points[i])
    print "\nIterations per game: {:.2f} +- {:.2f}".format(np.mean(iterations), np.std(iterations))
    print "Time out is reached {:.2f}% of the time"\
        .format(100*sum(float(x==MAX_ITER) for x in iterations)/len(iterations))