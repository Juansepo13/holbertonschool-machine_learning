#!/usr/bin/env python3
"""
    Calculates the mean and standard deviation of each feature
"""

import numpy as np


def normalization_constants(X):
    """
        Calculates the mean and standard deviation of each feature

        Args:
            X: numpy.ndarray of shape (m, nx) to normalize

        Returns:
            mean: mean of all features on X
            standard_deviation: standard deviation of all features on X
    """

    mean = np.mean(X, axis=0)
    standard_deviation = np.std(X, axis=0)

    return mean, standard_deviation
