#!/usr/bin/env python3
"""Module containing the function one_hot_encode whcih converts a numeric label
vector into a one-hot matrix."""

import numpy as np


def one_hot_encode(Y, classes):
    """[summary]

    Args:
        Y (numpy.ndarray): N-dimensional array with shape (m,) containing
        numeric class labels, where m is the number of examples.
        classes (int): The maximum number of classes found in Y

    Returns:
        A one-hot encoding of Y with shape (classes, m), or None on failure.
    """
    if not isinstance(Y, np.ndarray) or not(classes, int) or Y.ndim != 1:
        return None
    try:
        m = Y.shape[0]
        one_hot = np.zeros((m, classes))
        rows = np.arange(m)
        one_hot[rows, Y] = 1
    except Exception:
        return None

    return one_hot.T
