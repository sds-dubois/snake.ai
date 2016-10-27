import sys
import numpy as np
from controller import controller
from strategies import randomStrategy, greedyStrategy, smartGreedyStrategy

def simulate(n_simul, strategies, grid_size, candy_ratio = 1., max_iter = 500):
    results = dict((id, 0.) for id in xrange(len(strategies)))
    iterations = []
    for _ in xrange(n_simul):
        endState = controller(strategies, grid_size, candy_ratio = candy_ratio, max_iter = max_iter, verbose = 0)
        if len(endState.snakes) == 1:
            results[endState.snakes.keys()[0]] += 1. / n_simul
#        print endState.iter, ".",
        iterations.append(endState.iter)
    print "\n"
    return results, iterations


if __name__ ==  "__main__":
    MAX_ITER = 500

    if len(sys.argv) > 1:
        n_simul = int(sys.argv[1])
    else:
        n_simul = 500

    strategies = [randomStrategy, greedyStrategy, smartGreedyStrategy]
    results, iterations = simulate(n_simul, strategies, 20, max_iter = MAX_ITER)

    print "=======Results======="
    print "Max iteration:", MAX_ITER, "\n"
    for i in range(len(strategies)):
        print "\t Snake {} wins {:.2f}% of the games".format(i, results[i]*100)
    print "\nIterations per game: {:.2f} +- {:.2f}".format(np.mean(iterations), np.std(iterations))
    print "Time out is reached {:.2f}% of the time".format(sum(float(x==MAX_ITER) for x in iterations)/len(iterations))