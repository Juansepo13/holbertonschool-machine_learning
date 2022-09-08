#!/usr/bin/env python3
"""
Dropout gradient descent
"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Y is a one-hot numpy.ndarray of shape (classes, m)
    that contains the correct labels for the data
        classes is the number of classes
        m is the number of data points
    weights is a dictionary of the weights and biases of
    the neural network
    cache is a dictionary of the outputs and dropout masks
    of each layer of the neural network
    alpha is the learning rate
    keep_prob is the probability that a node will be kept
    L is the number of layers of the network
    All layers use thetanh activation function except the last,
    which uses the softmax activation function
    The weights of the network should be updated in place
    """
    m = Y.shape[1]
    weights_cop = weights.copy()
    for i in range(L, 0, -1):
        A = cache['A' + str(i)]
        if i == L:
            dz = A - Y
        else:
            dz = np.matmul(weights_cop['W' + str(i + 1)].T, dz) * (1 - (A * A))
            dz = dz * cache['D' + str(i)]
            dz = dz / keep_prob
        dw = 1 / m * np.matmul(dz, cache['A' + str(i - 1)].T)
        db = 1 / m * np.sum(dz, axis=1, keepdims=True)
        weights["W" + str(i)] = weights_cop['W' + str(i)] - alpha * dw
        weights["b" + str(i)] = weights_cop['b' + str(i)] - alpha * db
