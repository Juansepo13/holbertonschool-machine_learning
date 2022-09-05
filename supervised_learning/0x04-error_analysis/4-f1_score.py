#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def f1_score(confusion):
    """ function """
    ide = confusion * np.identity(confusion.shape[0])
    tp = np.sum(ide, axis=0)
    fp = np.sum(confusion - ide, axis=0)
    fn = np.sum(confusion - ide, axis=1)
    return tp / (tp + 0.5 * (fp + fn))
