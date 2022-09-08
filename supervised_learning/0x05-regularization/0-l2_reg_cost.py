#!/usr/bin/env python3
""" Cost with L2 regularization"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ Cost with L2 regularization"""
    l2_reg_cost = 0
    for layer in range(1, L + 1):
        l2_reg_cost += np.sum(np.square(weights["W" + str(layer)]))
    l2_cost = (1/m) * (lambtha/2) * l2_reg_cost
    return cost + l2_cost