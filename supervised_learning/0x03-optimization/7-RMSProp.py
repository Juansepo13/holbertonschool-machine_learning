#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """ rms prop """
    dW = (s * beta2) + (1 - beta2) * np.square(grad)
    out = var - ((alpha * grad) / (np.sqrt(dW) + epsilon))
    return out, dW
    