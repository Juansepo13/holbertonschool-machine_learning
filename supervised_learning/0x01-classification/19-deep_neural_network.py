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

    def forward_prop(self, X):
        """Function that calculates the forward propagation of the deep
            neural network.

        Args:
            X (numpy.ndarray): N-dimensional array with shape (nx, m) that
            contains the input data, where nx is the number of input features
            to the deep neural network and m is the number of examples.

        Returns:
            A (numpy.ndarray[(float)]): The activated output for the deep
            neural network.
        """
        A = X
        self.cache["A{}".format(0)] = X
        for l in range(1, self.L + 1):
            prev = A
            W = self.weights["W{}".format(l)]
            b = self.weights["b{}".format(l)]
            Z = np.matmul(W, A) + b
            A = 1/(1 + np.exp(-Z))  # Sigmoid
            self.__cache["A{}".format(l)] = A
        return A, self.cache

    def cost(self, Y, A):
        """Function that alculates the cost of the model using
            logistic regression.

        Args:
            Y (numpy.ndarray): N-dimensional array with shape (1, m) that
                contains the correct labels for the input data.
            A (numpy.ndarray): N-dimensioal array with shape (1, m) containing
                the activated output of the neuron for each example.
                Sometiems refered to as "y hat" a y with a "^" above it.

        Returns:
            The cost of the model.
        """
        m = Y.shape[1]
        loss_array = -((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A)))
        return np.sum(loss_array) / m
