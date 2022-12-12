#!/usr/bin/env python3
"""
File that contains the class RNNCell
"""

import numpy as np


class RNNCell:
    """
    Class RNNCell that represents a cell of a simple RNN
    """

    def __init__(self, i, h, o):
        """
        Constructor that initializes the public instance attributes
        Attributes:
            - i is the dimensionality of the data
            - h is the dimensionality of the hidden state
            - o is the dimensionality of the outputs
        Public instance attributes:
            - Wh: weights for the concatenated hidden state and input data
            - Wy: weights for the output
            - bh: bias for the concatenated hidden state and input data
            - by: bias for the output
        """
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        The output of the cell should use a softmax activation function
        Returns: h_next, y
            h_next is the next hidden state
            y is the output of the cell
        Method that performs forward propagation for one time step
        Arguments:
            - h_prev: numpy.ndarray of shape (m, h) containing the
                      previous hidden state
            - x_t: numpy.ndarray of shape (m, i) that contains the
                   data input for the cell
        Returns:
            - h_next, y
        """
        h_next = np.tanh(np.matmul(np.concatenate(
            (h_prev, x_t), axis=1), self.Wh) + self.bh)
        y = np.exp(np.matmul(h_next, self.Wy) + self.by) / \
            np.sum(np.exp(np.matmul(h_next, self.Wy) + self.by),
                   axis=1, keepdims=True)
        return h_next, y
