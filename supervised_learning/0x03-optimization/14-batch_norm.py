#!/usr/bin/env python3
"""
    module
"""
import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """ function """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    lay = tf.layers.Dense(n, kernel_initializer=init)(prev)
    mean, variance = tf.nn.moments(lay, axes=[0])
    gamma = tf.Variable(tf.constant(
        1.0, shape=[n]), trainable=True, name='gamma')
    beta = tf.Variable(tf.constant(
        0.0, shape=[n]), trainable=True, name='beta')
    out = tf.nn.batch_normalization(lay, mean, variance, beta, gamma, 1e-8)
    return activation(out)
