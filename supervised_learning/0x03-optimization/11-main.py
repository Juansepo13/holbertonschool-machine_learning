#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


learning_rate_decay = __import__('11-learning_rate_decay').learning_rate_decay

if __name__ == '__main__':
    alpha_init = 0.1
    for i in range(100):
        alpha = learning_rate_decay(alpha_init, 1, i, 10)
        print(alpha)
