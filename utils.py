import sys
import numpy as np

"""
Utils
"""

def softmax(x):
    """ Compute softmax values for each sets of scores in x """
    e_x = np.exp(x - np.max(x)) # subtract max(x) for stability
    return e_x / e_x.sum(axis = 0)

def add(tuple1, tuple2, mu = 1):
    """
    Return tuple1 + mu * tuple2.
    """
    return tuple([tuple1[i] + mu * tuple2[i] for i in xrange(len(tuple1))])

def mult(t, mu):
    return tuple([x * mu for x in t])

def dist(tuple1, tuple2):
    """Manhattan distance"""
    return abs(tuple1[0] - tuple2[0]) + abs(tuple1[1] - tuple2[1])

def norm1(tuple):
    """Norm 1"""
    return dist(tuple, tuple)

def rotate(p, dir):
    """Rotate position `p` in relative coordinates when snake has direction `dir`"""

    if dir == (0,-1):
        return mult(p, -1)
    elif dir == (1,0):
        return (-p[1], p[0])
    elif dir == (-1,0):
        return (p[1], - p[0])
    else:
        return p

def rotateBack(p, dir):
    """Rotate position `p` in aboslute coordinates when snake has direction `dir`"""

    if dir == (0,-1):
        return mult(p, -1)
    elif dir == (1,0):
        return (p[1], - p[0])
    elif dir == (-1,0):
        return (- p[1], p[0])
    else:
        return p

def isOnGrid(p, grid_size):
    """
    Check if position `p` is valid for the grid.
    """
    return p[0] >= 0 and p[1] >= 0 and p[0] < grid_size and p[1] < grid_size

def progressBar(iteration, n_total, size = 50, info = None):
    size = min(size, n_total)
    if iteration % (n_total/size) == 0:
        sys.stdout.write('\r')
        i = iteration*size/n_total
        if info is not None:
            sys.stdout.write("[<{}D-<{}] {}% | {}".format('='*i, ' '*(size-i), (100/size)*i, info))
        else:
            sys.stdout.write("[<{}D-<{}] {}%".format('='*i, ' '*(size-i), (100/size)*i))
        sys.stdout.flush()
    if iteration == n_total:
        print ""
