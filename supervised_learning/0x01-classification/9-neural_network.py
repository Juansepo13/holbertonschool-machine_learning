#!/usr/bin/env python3
"""Module containing the class NeuralNetwork which defines a neural network
with one hidden layer performing binary classification."""

import numpy as np


class NeuralNetwork():
    """Class that defines a neural network with one hidden layer performing
    binary classification."""

    def __init__(self, nx, nodes):
        """Innitilization function for NeuralNetwork class

        Args:
            nx (int): The number of input features
            nodes (int): The number of nodes found in the hidden layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        elif not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        elif nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__W2 = np.random.randn(1, nodes)
        self.__b1 = np.zeros((nodes, 1))
        self.__b2 = 0
        self.__A1 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter method for __W1"""
        return self.__W1

    @property
    def W2(self):
        """Getter method for __W2"""
        return self.__W2

    @property
    def b1(self):
        """Getter method for __b1"""
        return self.__b1

    @property
    def b2(self):
        """Getter method for __b2"""
        return self.__b2

    @property
    def A1(self):
        """Getter method for __A1"""
        return self.__A1

    @property
    def A2(self):
        """Getter method for __A2"""
        return self.__A2
