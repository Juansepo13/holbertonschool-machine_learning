#!/usr/bin/env python3
"""
Module for class Neuron
"""
import numpy as np


class Neuron:
    """
    Defines a single neuron performing binary classification
    """

    def __init__(self, nx):
        """ Class constructor """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ Getter function for W """
        return self.__W

    @property
    def b(self):
        """ Getter function for b """
        return self.__b

    @property
    def A(self):
        """ Getter function for A """
        return self.__A
