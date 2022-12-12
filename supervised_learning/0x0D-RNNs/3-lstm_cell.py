#!/usr/bin/env python3
"""
File that contains the class LSTMCell
"""

import numpy as np


class LSTMCell:
    """
    Class LSTMCell that represents an LSTM unit
    """

    def __init__(self, i, h, o):
        """
            Converts the weights and biases to a dictionary to
            represent the weights and biases of the cell
            Arguments:
                - i: dimensionality of the data
                - h: dimensionality of the hidden state
                - o: dimensionality of the outputs
            Public instance attributes:
                - Wf: weights for the forget gate
                - Wu: weights for the update gate
                - Wc: weights for the intermediate cell state
                - Wo: weights for the output gate
                - Wy: weights for the output
                - bf: bias for the forget gate
                - bu: bias for the update gate
                - bc: bias for the intermediate cell state
                - bo: bias for the output gate
                - by: bias for the output
        """
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """
        Method that performs the softmax
        Arguments:
            - x: numpy.ndarray of shape (t, m, i) that contains
                    the data to be transformed
                * t is the maximum number of time steps
                * m is the batch size for the data
                * i is the dimensionality of the data
        Returns:
            - softmax: numpy.ndarray of shape (t, m, i) containing
                        the softmax transformation
        """
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def sigmoid(self, x):
        """
        Method that performs the sigmoid
        Arguments:
            - x: numpy.ndarray of shape (t, m, i) that contains
                    the data to be transformed
                * t is the maximum number of time steps
                * m is the batch size for the data
                * i is the dimensionality of the data
        Returns:
            - sigmoid: numpy.ndarray of shape (t, m, i) containing
                        the sigmoid transformation
        """
        return 1 / (1 + np.exp(-x))

    def forward(self, h_prev, c_prev, x_t):
        """
        Method that performs forward propagation for one time step
        Arguments:
            - h_prev: numpy.ndarray of shape (m, h) containing the
                        previous hidden state
            - x_t: numpy.ndarray of shape (m, i) that contains the
                    data input for the cell
                * m is the batche size for the data
            - c_prev: numpy.ndarray of shape (m, h) containing the
                        previous cell state
        Returns: h_next, c_next, y
            - h_next: the next hidden state
            - c_next: the next cell state
            - y: the output of the cell
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        f = self.sigmoid(np.matmul(concat, self.Wf) + self.bf)
        u = self.sigmoid(np.matmul(concat, self.Wu) + self.bu)
        c = np.tanh(np.matmul(concat, self.Wc) + self.bc)
        c_next = f * c_prev + u * c
        o = self.sigmoid(np.matmul(concat, self.Wo) + self.bo)
        h_next = o * np.tanh(c_next)
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)
        return h_next, c_next, y
