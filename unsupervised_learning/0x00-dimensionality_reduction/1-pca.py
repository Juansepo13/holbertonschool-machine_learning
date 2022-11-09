#!/usr/bin/env python3
"""
0. PCA
"""
import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset
    Args:
        X: np.ndarray - shape (n, d)
            n: number of data points
            d: number of dimensions in each point
        ndim: new dimensionality of the transformed X
    Returns: T, a np.ndarray - shape (n, ndim)
        contains the transformed version of X
    """
    X_mean = X - np.mean(X, axis=0)
    u, s, vh = np.linalg.svd(X_mean)
    W = vh[:ndim].T
    T = np.matmul(X_mean, W)
    return T
