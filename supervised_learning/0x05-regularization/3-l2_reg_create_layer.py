#!/usr/bin/env python3
"""
3. Create a Layer with L2 Regularization
"""
import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a tensorflow layer that includes L2 reg:
    prev: tensor containing the output of the previous layer
    n: number of nodes the new layer should contain
    activation: activation function that should be used
    lambtha: L2 regularization parameter
    Returns: the output of the new layer
    """
    l2_reg = tf.keras.regularizers.L2(lambtha)
    weight = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=weight,
                            kernel_regularizer=l2_reg)(prev)
    return layer
