#!/usr/bin/env python3
"""
    module
"""
import tensorflow.compat.v1 as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """ tf func """
    return tf.train.RMSPropOptimizer(
        alpha,
        decay=beta2,
        epsilon=epsilon
    ).minimize(loss)
