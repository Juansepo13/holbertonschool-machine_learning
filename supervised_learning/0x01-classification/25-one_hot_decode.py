#!/usr/bin/env python3
"""Module that contains the function one_hot_decode that converts a one-hot
matrix into a vector of labels"""

import numpy as np


def one_hot_decode(one_hot):
    """ Function that converts a one-hot matrix into a vector of labels.

    Args:
        one_hot (numpy.ndarray):  A one-hot encoded numpy.ndarray with shape
        (classes, m), where classes is the maximum number of classes amd m is
        the number of examples.

    Returns:
        lables (numpy.ndarray): N-dimensional array with shape (m, ) containing
        the numeric labels for each example, or None on failure.
    """
    if not isinstance(one_hot, np.ndarray) or one_hot.ndim != 2:
        return None

    lables = one_hot.argmax(0)

    return lables
