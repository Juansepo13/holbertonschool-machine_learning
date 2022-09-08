#!/usr/bin/env python3
'''Gradient Descent with Dropout module'''
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    '''updates the weights of a neural network with Dropout regularization
    using gradient descent
    Args:
        Y is a one-hot numpy.ndarray of shape (classes, m) that contains the
        correct labels for the data
          classes is the number of classes
          m is the number of data points
        weights is a dictionary of the weights and biases of the neural network
        cache is a dictionary of the outputs and dropout masks of each layer of
              the neural network
        alpha is the learning rate
        keep_prob is the probability that a node will be kept
        L is the number of layers of the network
    Returns: Nothing. The weights and biases of the network should be updated
             in place
    Important: All layers except the last uses the tanh activation function
               The last layer uses the softmax activation function
    '''
    m = Y.shape[1]
    len_cache = L + 1

    # learning for the last layer:
    Al = cache['A{}'.format(len_cache - 1)]  # last A
    A_prev = cache['A{}'.format(len_cache - 2)]  # pre last A
    dZl = Al - Y  # last dZ
    dWl = np.matmul(dZl, A_prev.T) / m  # last dW, shape (1, nodes)
    dbl = (1 / m) * np.sum(dZl, axis=1, keepdims=True)
    Wl_str = 'W{}'.format(len_cache - 1)
    Wl = weights[Wl_str]  # last W
    # last layer W learning:
    weights[Wl_str] = Wl - alpha * dWl
    bl_str = 'b{}'.format(len_cache - 1)
    bl = weights[bl_str]  # last b
    weights[bl_str] = bl - alpha * dbl  # last layer b learning

    #  next: learning for the rest of the layers:
    dZ = dZl
    W_next = Wl
    for i in reversed(range(1, len_cache - 1)):
        A = cache['A{}'.format(i)]
        D = cache['D{}'.format(i)]
        A_prev = cache['A{}'.format(i - 1)]
        # with tanh activation and dropout:
        dZ = np.matmul(W_next.T, dZ) * (1 - A ** 2) * D / keep_prob
        dW = (1 / m) * (np.matmul(dZ, A_prev.T))
        db = np.sum(dZ, axis=1, keepdims=True) / m
        W_c_str = 'W{}'.format(i)
        W_c = weights[W_c_str]  # current W
        b_c_str = 'b{}'.format(i)
        b_c = weights[b_c_str]  # current b
        weights[W_c_str] = W_c - alpha * dW
        weights[b_c_str] = b_c - alpha * db
        W_next = W_c
