#!/usr/bin/env python3
"""
    module
"""


def moving_average(data, beta):
    """ moving """
    out = []
    last = 0
    for idx, val in enumerate(data):
        last = beta * last + (1 - beta) * val
        out.append(last / (1 - beta ** (idx + 1)))
    return out
