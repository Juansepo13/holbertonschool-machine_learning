#!/usr/bin/env python3
"""
    module
"""

def update_variables_momentum(alpha, beta1, var, grad, v):
    """update vars"""
    dw = (v * beta1) + ((1 - beta1) * grad)
    return var - (dw * alpha), dw
