#!/usr/bin/env python3
"""
2. Marginal Probability
"""
import numpy as np


def marginal(x, n, P, Pr):
    """
    Calculates the marginal probability of obtaining the data
    Args:
        x: number of patients that develop severe side effects
        n: total number of patients observed
        P: 1D np.ndarray containing the various hypothetical probabilities
            of developing severe side effects
        Pr: 1D np.ndarray containing the prior beliefs of P
    Returns: 1D np.ndarray containing the intersection of obtaining
        x and n with each probability in P, respectively
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not (isinstance(P, np.ndarray) and len(P.shape) == 1):
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    for i in range(P.shape[0]):
        if P[i] > 1 or P[i] < 0:
            raise ValueError("All values in P must be in the range [0, 1]")
        if Pr[i] > 1 or Pr[i] < 0:
            raise ValueError("All values in Pr must be in the range [0, 1]")
    if np.isclose([np.sum(Pr)], [1]) == [False]:
        raise ValueError("Pr must sum to 1")

    factorial = np.math.factorial
    llhood = (factorial(n) / (factorial(x) * factorial(n - x)) * (P ** x) *
              ((1 - P) ** (n - x)))
    intersection = llhood * Pr
    return np.sum(intersection)
