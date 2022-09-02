#!/usr/bin/env python3
"""
    module
"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ learning rate decay """
    out = alpha * (1 / (1 + (decay_rate * (global_step % decay_step))))
    return out
