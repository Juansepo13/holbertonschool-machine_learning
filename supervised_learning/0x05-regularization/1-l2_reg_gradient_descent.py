#!/usr/bin/env python3
""" Gradient descent with L2 regularization """
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ Gradient descent with L2 regularization """
    m = Y.shape[1]
    # cost
    dz = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        a = 'A' + str(i - 1)
        w = 'W' + str(i)
        b = 'b' + str(i)
        A = cache[a]
        dw = (1 / m) * np.matmul(dz, np.transpose(A)) + (lambtha / m) * \
            weights[w]
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        dz = np.matmul(np.transpose(weights[w]), dz) * (1 - A**2)
        weights[w] = weights[w] - (alpha * dw)
        weights[b] = weights[b] - (alpha * db)
