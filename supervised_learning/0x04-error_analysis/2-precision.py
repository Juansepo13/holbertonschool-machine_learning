#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def precision(confusion):
    """ function """
    iden = np.identity(confusion.shape[0])
    out = np.sum(iden * confusion, axis=0)
    out /= np.sum(confusion, axis=0)
    return out
