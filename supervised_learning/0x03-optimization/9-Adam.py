#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """ adam """
    vdw = (beta1 * v) + ((1 - beta1) * grad)
    sdw = (beta2 * s) + ((1 - beta2) * np.square(grad))
    c_vdw = vdw / (1 - (beta1 ** t))
    c_sdw = sdw / (1 - (beta2 ** t))
    out = var - alpha * (c_vdw / (np.sqrt(c_sdw) + epsilon))
    return out, vdw, sdw
