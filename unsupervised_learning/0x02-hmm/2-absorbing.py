#!/usr/bin/env python3
"""Module for the function
"""

import numpy as np


def absorbing(P):
    """Determines if a markov chain is absorbing
    Args.
        P: a square 2D numpy.ndarray of shape (n, n) representing the
        standard transition matrix
            -P[i, j] is the probability of transitioning from state i to
             state j
            -n is the number of states in the markov chain
    Returns.
        True if it is absorbing, or False on failure
    """

    if type(P) is not np.ndarray or len(P.shape) != 2:
        return False

    rows, columns = P.shape

    if rows != columns:
        return False

    if np.sum(P, axis=1).all() != 1:
        return False

    D = np.diagonal(P)

    if np.all(D == 1):
        return True

    if not np.any(D == 1):
        return False

    count = np.count_nonzero(D == 1)
    Q = P[count:, count:]
    Id = np.eye(Q.shape[0])

    try:
        if (np.any(np.linalg.inv(Id - Q))):
            return True
    except np.linalg.LinAlgError:
        return False
