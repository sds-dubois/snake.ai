import sys
from controller import controller
from strategies import greedy, smartGreedy

def simulate(n_simul, strategies, grid_size, candy_ratio = 1., max_iter = 500):
    results = dict((id, 0.) for id in xrange(len(strategies)))
    for _ in xrange(n_simul):
        endState = controller(strategies, grid_size, candy_ratio = candy_ratio, max_iter = max_iter, verbose = 0)
        if len(endState.snakes) == 1:
            results[endState.snakes.keys()[0]] += 1. / n_simul
        print endState.iter, ".",
    print "\n"
    return results


if __name__ ==  "__main__":
    if len(sys.argv) > 1:
        n_simul = int(sys.argv[1])
    else:
        n_simul = 100

    # results = simulate(n_simul, [greedy, smartGreedy, smartGreedy], 20, max_iter = 500)
    results = simulate(n_simul, [greedy, smartGreedy], 20, max_iter = 500)

    print results