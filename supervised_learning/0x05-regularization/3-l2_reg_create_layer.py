#!/usr/bin/env python3
""" Layer with tensorflow including L2 regularization """
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """ Layer with tensorflow including L2 regularization """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    regularizer = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(n, activation, name='layer',
                            kernel_initializer=init,
                            kernel_regularizer=regularizer)
    out = layer(prev)
    return out
