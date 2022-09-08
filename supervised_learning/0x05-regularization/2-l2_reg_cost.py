#!/usr/bin/env python3
""" Cost L2 regularization """
import tensorflow as tf


def l2_reg_cost(cost):
    """ Cost L2 regularization """
    loss = tf.losses.get_regularization_losses()
    return cost + loss
