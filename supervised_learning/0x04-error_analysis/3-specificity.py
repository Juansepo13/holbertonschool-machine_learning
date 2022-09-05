#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def specificity(confusion):
    """ function """
    total = np.sum(confusion)
    ide = confusion * np.identity(confusion.shape[0])
    tn = total - (np.sum(confusion - ide, axis=0) + np.sum(confusion, axis=1))
    fp = np.sum(confusion - ide, axis=0)
    return tn / (fp + tn)
