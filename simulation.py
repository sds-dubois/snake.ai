import sys
import numpy as np
from time import sleep, time

import config
from utils import progressBar
from controller import controller
from strategies import randomStrategy, greedyStrategy, smartGreedyStrategy, opportunistStrategy
from rl import rl_strategy, load_rl_strategy
from features import FeatureExtractor
from minimax import AlphaBetaAgent, ExpectimaxAgent, greedyEvaluationFunction

def simulate(n_simul, strategies, grid_size, candy_ratio = 1., max_iter = 500):
    print "Simulations"
    wins = dict((id, 0.) for id in xrange(len(strategies)))
    points = dict((id, []) for id in xrange(len(strategies)))
    scores = dict((id, []) for id in xrange(len(strategies)))

    iterations = []
    for it in xrange(n_simul):
        progressBar(it, n_simul)
        endState = controller(strategies, grid_size, candy_ratio = candy_ratio, max_iter = max_iter, verbose = 0)
        if len(endState.snakes) == 1:
            wins[endState.snakes.keys()[0]] += 1. / n_simul
            points[endState.snakes.keys()[0]].append(endState.snakes.values()[0].points)

        for id in xrange(len(strategies)):
            scores[id].append(endState.scores[id])

        iterations.append(endState.iter)
    progressBar(n_simul, n_simul)
    points = dict((id, sum(val)/len(val)) for id,val in points.iteritems())
    return wins, points, scores, iterations


if __name__ ==  "__main__":
    MAX_ITER = 1000

    if len(sys.argv) > 1:
        n_simul = int(sys.argv[1])
    else:
        n_simul = 1000

    print "Simulation config:", ["{} = {}".format(k,v) for k,v in config.__dict__.iteritems() if not k.startswith('__')]

    strategies = config.opponents
    if config.agent == "RL":
        featureExtractor = FeatureExtractor(len(config.opponents), config.grid_size, radius_ = config.radius)
        if len(sys.argv) > 2 and sys.argv[2] == "load":
            print "Loading weights.."
            rlStrategy = load_rl_strategy(config.filename + ".p", config.opponents, featureExtractor, config.discount, q_type = config.rl_type)
        else:
            rlStrategy = rl_strategy(config.opponents, featureExtractor, config.discount, config.grid_size, q_type = config.rl_type, lambda_ = config.lambda_, num_trials = config.num_trials, max_iter = config.max_iter, filename = config.filename + ".p")
        strategies.append(rlStrategy)
    elif config.agent == "AlphaBeta":
        agent = AlphaBetaAgent(depth = config.depth, evalFn = config.evalFn)
        strategies.append(agent.getAction)
    elif config.agent == "ExpectimaxAgent":
        agent = ExpectimaxAgent(depth = config.depth, evalFn = config.evalFn)
        strategies.append(agent.getAction)

    start = time()
    wins, points, scores, iterations = simulate(n_simul, strategies, config.grid_size, max_iter = MAX_ITER)
    tot_time = time() - start

    with open("experiments/{}_{}_{}.txt".format(config.filename, "-".join([s.__name__ for s in strategies]).replace("<lambda>", "rl"), config.comment), "wb") as fout:
        print >> fout, "\n\n=======Results======="
        print >> fout, "Run {} simulations".format(n_simul)
        print >> fout, "Max iteration:", MAX_ITER, "\n"

        for i in range(len(strategies)):
            print >> fout, "\t Snake {} wins {:.2f}% of the games, with {:.2f} points on average".format(i, wins[i]*100, points[i])
            print "\t Snake {} wins {:.2f}% of the games, with {:.2f} points on average".format(i, wins[i]*100, points[i])
        print >> fout, "\nScores"
        print "\nScores"
        for i in range(len(strategies)):
            print >> fout, "\t Snake {}: avg score = {:.2f}, finishes with {:.2f} points on average".format(i, np.mean([p/r for r,p in scores[i]]), np.mean([p for r,p in scores[i]]))
            print "\t Snake {}: avg score = {:.2f}, finishes with {:.2f} points on average".format(i, np.mean([p/r for r,p in scores[i]]), np.mean([p for r,p in scores[i]]))
        print >> fout, "\nIterations per game: {:.2f} +- {:.2f}".format(np.mean(iterations), np.std(iterations))
        print >> fout, "Time out is reached {:.2f}% of the time"\
            .format(100*sum(float(x==MAX_ITER) for x in iterations)/len(iterations))
        print >> fout, "Simulations took {} sec on avg".format(tot_time / n_simul)

        print >> fout, "\n\nParams"
        print >> fout, "\n".join(
            ["{} = {}".format(k,v) for k,v in config.__dict__.iteritems() if not k.startswith('__')]
        )