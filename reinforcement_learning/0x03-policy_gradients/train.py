"""
File
"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    Function that implements a full training.
    Arguments:
        - env is the openAI environment instance.
        - nb_episodes is the total number of episodes to train over.
        - alpha is the learning rate.
        - gamma is the discount factor.
    Returns:
        - all values of the score (sum of all rewards during one episode loop)
    """
    nA = env.action_space.n
    nS = env.observation_space.shape[0]
    weight = np.random.rand(nS, nA)
    scores = []
    for i in range(nb_episodes):
        state = env.reset()
        done = False
        score = 0
        while not done:
            action, grad = policy_gradient(state, weight)
            state, reward, done, info = env.step(action)
            weight += alpha * grad * reward
            score += reward
        scores.append(score)
    return scores
