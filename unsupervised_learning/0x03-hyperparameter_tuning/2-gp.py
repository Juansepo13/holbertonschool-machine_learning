#!/usr/bin/env python3
"""
2. Update Gaussian Process
"""
import numpy as np


class GaussianProcess:
    """
    Represents a noiseless 1D Gaussian process
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Class constructor
        Args:
            X_init: np.ndarray - (t, 1) - inputs already sampled with the
                black-box function
            Y_init: np.ndarray - (t, 1) - outputs of the black-box function
                for each input in X_init
            l: length parameter for the kernel
            sigma_f: standard deviation given to the output of the
                black-box function
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X1=X_init, X2=X_init)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix between two matrices
        Args:
            X1: np.ndarray - (m, 1)
            X2: np.ndarray - (n, 1)
        kernel should use the Radial Basis Function (RBF)
        Returns: covariance kernel matrix as a np.ndarray - (m, n)
        """
        cov = np.exp(-((X1 - X2.T) ** 2) / (2 * (self.l ** 2)))
        return (self.sigma_f ** 2) * cov

    def predict(self, X_s):
        """
        Predicts the mean and standard deviation
        Args:
            X_s: np.ndarray - (s, 1) - points whose mean and standard
                deviation should be calculated

        Returns: mu, sigma
        """
        s = X_s.size

        cov = self.kernel(self.X, X_s)
        solution = np.linalg.solve(self.K, cov).T
        cov2 = self.kernel(X_s, X_s)
        mu = solution @ self.Y
        sigma = cov2 - (solution @ cov)

        return mu.reshape(s, ), np.diag(sigma)

    def update(self, X_new, Y_new):
        """
        Updates a Gaussian Process
        Args:
            X_new: np.ndarray - (1,) - new sample point
            Y_new: np.ndarray - (1,) - new sample function value
        """
        self.X = np.append(self.X, X_new).reshape(-1, 1)
        self.Y = np.append(self.Y, Y_new).reshape(-1, 1)
        self.K = self.kernel(self.X, self.X)
