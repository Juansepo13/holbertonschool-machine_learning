#!/usr/bin/env python3
"""
    module
"""
import tensorflow.compat.v1 as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ tf func """
    return tf.compat.v1.train.inverse_time_decay(
        alpha, global_step, decay_step, decay_rate, staircase=True
    )
