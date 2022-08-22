#!/usr/bin/env python3
import numpy as np
"""Write a class Neuron that defines a single neuron performing binary classification"""


class Neuron:
    """neuron performing binary classification"""
    def __init__(self, nx):
        """class constructor"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.b = 0
        self.A = 0
        self.W = np.random.randn(1, nx)
