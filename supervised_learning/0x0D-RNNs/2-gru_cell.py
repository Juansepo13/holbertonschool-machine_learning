#!/usr/bin/env python3
"""
File that contains the class RNNCell
"""

import numpy as np


class GRUCell:
    """
    Class GRUCell that represents a gated recurrent unit
    """

    def __init__(self, i, h, o):
        """
        Constructor that initializes the public instance attributes
        Attributes:
            - i is the dimensionality of the data
            - h is the dimensionality of the hidden state
            - o is the dimensionality of the outputs
        Public instance attributes:
            - Wz: weights for the update gate
            - Wr: weights for the reset gate
            - Wh: weights for the intermediate hidden state
            - Wy: weights for the output
            - bz: bias for the update gate
            - br: bias for the reset gate
            - bh: bias for the intermediate hidden state
            - by: bias for the output
        """
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Method that performs forward propagation for one time step
        Arguments:
            - h_prev: numpy.ndarray of shape (m, h) containing the
                        previous hidden state
            - x_t: numpy.ndarray of shape (m, i) that contains the
                    data input for the cell
        Returns:
            - h_next is the next hidden state
            - y is the output of the cell
        """
        z = np.exp(np.matmul(np.concatenate((h_prev, x_t), axis=1),
                             self.Wz) + self.bz) / \
            (np.exp(np.matmul(np.concatenate((h_prev, x_t), axis=1),
                              self.Wz) + self.bz) + 1)
        r = np.exp(np.matmul(np.concatenate((h_prev, x_t), axis=1),
                             self.Wr) + self.br) / \
            (np.exp(np.matmul(np.concatenate((h_prev, x_t), axis=1),
                              self.Wr) + self.br) + 1)
        h = np.tanh(np.matmul(np.concatenate((r * h_prev, x_t), axis=1),
                              self.Wh) + self.bh)
        h_next = (1 - z) * h_prev + z * h
        y = np.exp(np.matmul(h_next, self.Wy) + self.by) / \
            np.sum(np.exp(np.matmul(h_next, self.Wy) + self.by),
                   axis=1, keepdims=True)
        return h_next, y
