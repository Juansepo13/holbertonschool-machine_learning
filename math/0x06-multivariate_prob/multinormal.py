#!/usr/bin/env python3
"""Module contains MultiNormal Class."""


import numpy as np


class MultiNormal():
    """
       Class which represents multivariate normal
        distribution.
    """

    def __init__(self, data):
        """
        Class Constructor
        Args:
            data: numpy.ndarray of shape (d, n)
                n: Number of data points.
                d: Number of dimensions in each data point.
        Public Attributes:
            mean: numpy.ndarray of shape (d, 1)
            cov: numpy.ndarray of shape (d, d
        """
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        self.mean, self.cov = self.mean_cov(data.T)