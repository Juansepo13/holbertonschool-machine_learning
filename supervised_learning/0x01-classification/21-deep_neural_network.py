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

    def evaluate(self, X, Y):
        """Function that valuates the deep neural network’s predictions.

        Args:
            X (numpy.ndarray): N-dimensioal array with shape (nx, m) that
                contains the input data, where nx is the number of input
                features to the neuron and m is the number of examples.
            Y (numpy.ndarray): N-dimensioal array with shape (1, m) that
                contains the correct labels for the input data.

        Returns:
            A (numpy.ndarray): The neuron’s prediction. The predictions shape
                will be (1, m), containing the predicted labels for each
                example.
            cost (float): The cost of the network.
        """
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        A = np.where(A >= 0.5, 1, 0)
        return A, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Function that calculates one pass of gradient descent on the neural
        network.

        Args:
            Y (numpy.ndarray): N-dimensioal array with shape (1, m) that
                contains the correct labels for the input data.
            cache (dictionary): Dictionary containing all the intermediary
                values of the network.
            alpha (float, optional): The learning rate. Defaults to 0.05.
        """
        m = Y.shape[1]

        for i in range(self.L, 0, -1):
            Ai = cache["A{}".format(i)]
            Ai_next = cache["A{}".format(i - 1)]

            if i == self.L:
                dZi = (Ai - Y)
            else:
                dZi = dAi_next * (Ai * (1 - Ai))  # σ'((z)) = σ(z)(1-σ(z))
            dWi = np.matmul(dZi, Ai_next.T) / m
            dbi = np.sum(dZi, axis=1, keepdims=True) / m

            dZ_prev = dZi
            Wi = self.weights["W{}".format(i)]
            dAi_next = np.matmul(Wi.T, dZi)

            self.__weights["W{}".format(i)] -= (alpha * dWi)
            self.__weights["b{}".format(i)] -= (alpha * dbi)
