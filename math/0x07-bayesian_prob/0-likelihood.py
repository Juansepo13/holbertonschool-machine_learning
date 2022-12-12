#!/usr/bin/env python3
"""
0. Likelihood
"""
import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining this data
        given various hypothetical probabilities
    Args:
        x: number of patients that develop severe side effects
        n: total number of patients observed
        P: 1D np.ndarray containing the various hypothetical probabilities
            of developing severe side effects
    Returns: 1D np.ndarray containing the likelihood of obtaining
        the data, x and n, for each probability in P, respectively
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
    if any([True if (x > 1 or x < 0) else False for x in P]):
        raise ValueError("All values in P must be in the range [0, 1]")
    # Binomal distribution likelihood
    # pr(x|n, p) = (n! / x!(n - x)!)* (p**x) * (1 - p)**(n - x)
    factorial = np.math.factorial
    llhood = (factorial(n) / (factorial(x) * factorial(n - x)) * (P ** x) *
              ((1 - P) ** (n - x)))
    return llhood
