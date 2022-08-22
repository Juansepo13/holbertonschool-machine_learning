#!/usr/bin/env python3
"""Module containing the class Neuron which defines a single neuron performing
binary classification"""

import numpy as np


class Neuron():
    """Class which defines a single neuron performing binary classification
    """
    def __init__(self, nx):
        """Initizilation function for Neuron

        Args:
            nx (int): The number of input features to the neuron
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        else:
            self.__W = np.random.randn(1, nx)
            self.__b = 0
            self.__A = 0

    @property
    def W(self):
        """Getter for private instance atribute W.

        Returns:
            [float]: The weights vector for the neuron.
        """
        return self.__W

    @property
    def b(self):
        """Getter for private instance atribute b

        Returns:
            [float]: The bias for the neuron.
        """
        return self.__b

    @property
    def A(self):
        """Getter for private instance atribute

        Returns:
            [numpy.ndarray]: The activated output of the neuron.
        """
        return self.__A
