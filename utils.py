"""
Utils
"""

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

