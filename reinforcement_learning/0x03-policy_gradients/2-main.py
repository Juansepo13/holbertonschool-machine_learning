#!/usr/bin/env python3
"""
Main file
"""
import gym
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from train import train

env = gym.make('CartPole-v1')

scores = train(env, 10000)

plt.plot(np.arange(len(scores)), scores)
plt.show()
plt.savefig('2-train.png')

env.close()
