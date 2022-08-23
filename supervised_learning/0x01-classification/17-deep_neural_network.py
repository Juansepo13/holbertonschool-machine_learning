#!/usr/bin/env python3
"""Module containing the class DeepNeuralNetwork which defines a deep neural
network performing binary classification."""

import numpy as np


class DeepNeuralNetwork():
    """Class that defines a deep neural network performing binary
    classification."""

    def __init__(self, nx, layers):
        """Class constructor

        Args:
            nx (int): The number of input features
            layers (list[int]): List representing the number of nodes in each
                layer of the network
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        elif not isinstance(layers, list) or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        weights = {}
        prev = nx
        for i, l in enumerate(layers, 1):
            if not isinstance(l, int) or l < 1:
                raise TypeError("layers must be a list of positive integers")
            weights["b{}".format(i)] = np.zeros((l, 1))
            weights["W{}".format(i)] = (np.random.randn(l, prev) *
                                        np.sqrt(2 / prev))
            prev = l
        self.__weights = weights

    @property
    def L(self):
        """Getter for self.__L"""
        return self.__L

    @property
    def cache(self):
        """Getter for self.__cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for self.__weights"""
        return self.__weights
