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

        self.W1 = np.random.randn(nodes, nx)
        self.W2 = np.random.randn(1, nodes)
        self.b1 = np.zeros((nodes, 1))
        self.b2 = 0
        self.A1 = 0
        self.A2 = 0
