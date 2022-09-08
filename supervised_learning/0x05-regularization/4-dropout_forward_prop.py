#!/usr/bin/env python3
"""
Dropout regularization in forward prop
"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    * X is a numpy.ndarray of shape (nx, m) containing the input
         data for the network
    * nx is the number of input features
    * m is the number of data points
    * weights is a dictionary of the weights and biases of the neural network
    * L the number of layers in the network
    * keep_prob is the probability that a node will be kept
    * All layers except the last should use the tanh activation function
    * The last layer should use the softmax activation function
    * Returns: a dictionary containing the outputs of each layer and the
            dropout mask used on each layer (see example for format)
    """
    cache = {}
    cache["A0"] = X

    for i in range(1, L + 1):
        key_w = "W{}".format(i)
        key_b = "b{}".format(i)
        key_A = "A{}".format(i - 1)

        Z = np.matmul(weights[key_w], cache[key_A])
        Z = Z + weights[key_b]

        if i != L:
            D = np.random.binomial(1, keep_prob, size=Z.shape)
            A = (D / keep_prob) * np.tanh(Z)
            cache["D{}".format(i)] = D
        else:
            t = np.exp(Z)
            A = t / np.sum(t, axis=0, keepdims=True)

        cache["A{}".format(i)] = A

    return cache
