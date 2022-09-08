#!/usr/bin/env python3
"""
2. L2 Regularization Cost
"""
import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    """
    Calculates the cost of a neural network with L2 regularization
    cost: tensor containing the cost without L2 reg
    Returns: a tensor containing the cost with L2 regularization
    """
    return cost + tf.losses.get_regularization_losses()
