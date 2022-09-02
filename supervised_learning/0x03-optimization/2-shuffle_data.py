#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def shuffle_data(X, Y):
    """ shuffle data """
    idx = np.random.permutation(X.shape[0])
    return X[idx], Y[idx]
