#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def sensitivity(confusion):
    """ function """
    iden = np.identity(confusion.shape[0])
    out = np.sum(iden * confusion, axis=1)
    out /= np.sum(confusion, axis=1)
    return out
