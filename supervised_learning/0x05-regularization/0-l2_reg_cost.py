#!/usr/bin/env python3
""" Cost with L2 regularization"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ Cost with L2 regularization"""
    w_norm = 0
    for i in range(1, L + 1):
        w_norm += np.linalg.norm(weights['W' + str(i)])
    L2 = cost + (lambtha / (2 * m) * w_norm)
    return L2
