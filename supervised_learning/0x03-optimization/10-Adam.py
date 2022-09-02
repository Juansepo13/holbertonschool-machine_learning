#!/usr/bin/env python3
"""
    module
"""
import tensorflow.compat.v1 as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """ tf func """
    return tf.train.AdamOptimizer(
        alpha,
        beta1,
        beta2,
        epsilon
    ).minimize(loss)
