"""
Utils
"""

def add(tuple1, tuple2, lambda = 1):
    """
    Return tuple1 + lambda * tuple2.
    """
    return tuple([tuple1[i] + lambda * tuple2[i] for i in xrange(len(tuple1))])

def mult(t, lambda):
    return tuple([x * lambda for x in t])
