"""
File
"""
import numpy as np


def policy(matrix, weight):
    """
    Function that computes to policy with a weight of a matrix.
    Arguments:
     - matrix is a numpy.ndarray of shape (s, a) containing the
        policy network output.
     - weight is a numpy.ndarray of shape (s,) containing the
        weight for the matrix.
    """
    z = np.dot(matrix, weight)
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


def policy_gradient(state, weight):
    """
    Function that computes the Monte-Carlo policy gradient based on a state and a weight matrix.
    Arguments:
     - state is a numpy.ndarray of shape (s,) containing the current state.
     - weight is a numpy.ndarray of shape (s, a) containing the weight matrix.
    Returns:
     - The action and the gradient (in this order).
    """
    p = policy(state, weight)
    action = np.random.choice(len(p[0]), p=p[0])
    dsoftmax = p.copy()
    dsoftmax[0, action] -= 1
    dlog = dsoftmax / p
    grad = np.outer(state.T, dlog)
    return action, grad
