#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """ function """
    return np.dot(labels.T, logits)
