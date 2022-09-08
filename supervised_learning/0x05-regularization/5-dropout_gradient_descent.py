#!/usr/bin/env python3
"""
DropOut Regularization
"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Updates the weights and biases of a neural network using gradient
       descent with dropout regularization.
    Args:
        Y (np.ndarray): one-hot matrix of shape (classes, m) that contains
                        the correct labels for the data.
        weights (dict): the weights and biases of the neural network.
        cache (dict): the outputs of each layer of the neural network.
        alpha (float):  learning rate.
        keep_prob (float): probability that a node will be kept.
        L (int): number of layers of the network.
    Returns:
        dict: weights and biases of the network should be updated in place.
    """
    m = Y.shape[1]
    dz = {}
    dW = {}
    db = {}
    da = {}
    for la in reversed(range(1, L + 1)):
        A = cache["A{}".format(la)]
        A_prev = cache["A{}".format(la - 1)]

        # 3
        if la == L:
            kdz = "dz{}".format(la)
            kdW = "dW{}".format(la)
            kdb = "db{}".format(la)

            dz[kdz] = A - Y
            dW[kdW] = np.matmul(dz[kdz], A_prev.T) / m
            db[kdb] = dz[kdz].sum(axis=1, keepdims=True) / m

        else:
            # 2 - 1
            kdz_n = "dz{}".format(la + 1)
            kdz_c = "dz{}".format(la)
            kdW_n = "dW{}".format(la + 1)
            kdW = "dW{}".format(la)
            kdb_n = "db{}".format(la + 1)
            kdb = "db{}".format(la)
            kda = "da{}".format(la)
            kW = 'W{}'.format(la + 1)
            kb = 'b{}'.format(la + 1)
            kd = 'D{}'.format(la)

            W = weights[kW]
            D = cache[kd]

            da[kda] = np.matmul(W.T, dz[kdz_n])
            da[kda] *= D
            da[kda] /= keep_prob

            dz[kdz_c] = da[kda] * (1 - (A * A))
            dW[kdW] = np.matmul(dz[kdz_c], A_prev.T) / m
            db[kdb] = dz[kdz_c].sum(axis=1, keepdims=True) / m

            weights[kW] -= alpha * dW[kdW_n]
            weights[kb] -= alpha * db[kdb_n]

            if la == 1:
                weights['W1'] -= alpha * dW['dW1']
                weights['b1'] -= alpha * db['db1']
