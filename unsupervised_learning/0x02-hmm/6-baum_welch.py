#!/usr/bin/env python3
""" The Baum-Welch Algorithm """
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """ performs the forward algorithm for a hidden markov model.
        Args:
            Observation: (numpy.ndarray) contains the index of the observation.
            Emission: (numpy.ndarray) containing the emission probability of a
                      specific observation given a hidden state.
            Transition: (numpy.ndarray) containing the transition probabilities
            Initial: (numpy.ndarray) containing the probability of starting in
                     a particular hidden state.
        Returns:
            P, F, or None, None on failure.
            P: (float) likelihood of the observations given the model.
            F: (numpy.ndarray) containing the forward path probabilities.
    """
    if type(Observation) != np.ndarray or len(Observation.shape) != 1:
        return None, None
    if type(Emission) != np.ndarray or len(Emission.shape) != 2:
        return None, None
    if type(Transition) != np.ndarray or len(Transition.shape) != 2:
        return None, None
    if type(Initial) != np.ndarray or len(Initial.shape) != 2:
        return None, None
    T = Observation.shape[0]
    N, M = Emission.shape
    if Transition.shape[0] != N or Transition.shape[0] != Transition.shape[1]:
        return None, None
    if Initial.shape[0] != N or Initial.shape[1] != 1:
        return None, None
    F = np.zeros((N, T))
    # for i in range(N):
    #    F[i, 0] = Initial[i, 0] * Emission[i, Observation[0]]
    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]
    for t in range(1, T):
        for j in range(N):
            F[j, t] = np.sum(Transition[:, j] * F[:, t - 1] *
                             Emission[j, Observation[t]])
    P = np.sum(F[:, T - 1])

    return P, F


def backward(Observation, Emission, Transition, Initial):
    """ performs the backward algorithm for a hidden markov model.
        Args:
            Observation: (numpy.ndarray) contains the index of the observation.
            Emission: (numpy.ndarray) containing the emission probability of a
                      specific observation given a hidden state.
            Transition: (numpy.ndarray) containing the transition probabilities
            Initial: (numpy.ndarray) containing the probability of starting in
                     a particular hidden state.
        Returns:
            P, B, or None, None on failure.
            P: (float) likelihood of the observations given the model.
            B: (numpy.ndarray) containing the backward path probabilities.
    """
    if type(Observation) != np.ndarray or len(Observation.shape) != 1:
        return None, None
    if type(Emission) != np.ndarray or len(Emission.shape) != 2:
        return None, None
    if type(Transition) != np.ndarray or len(Transition.shape) != 2:
        return None, None
    if type(Initial) != np.ndarray or len(Initial.shape) != 2:
        return None, None
    T = Observation.shape[0]
    N, M = Emission.shape
    if Transition.shape[0] != N or Transition.shape[0] != Transition.shape[1]:
        return None, None
    if Initial.shape[0] != N or Initial.shape[1] != 1:
        return None, None
    B = np.zeros((N, T))
    # for i in range(N):
    #     B[i, T - 1] = 1
    B[:, T - 1] = 1
    for t in list(range(T - 1))[::-1]:
        for j in range(N):
            B[j, t] = np.sum(B[:, t + 1] *
                             Transition[j, :] *
                             Emission[:, Observation[t + 1]])
    P = np.sum(Emission[:, Observation[0]] * Initial[:, 0] * B[:, 0])

    return P, B


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """ performs the Baum-Welch algorithm for a hidden markov model.
        Args:
            Observation: (numpy.ndarray) contains the index of the observation.
            Emission: (numpy.ndarray) containing the emission probability of a
                      specific observation given a hidden state.
            Transition: (numpy.ndarray) containing the transition probabilities
            Initial: (numpy.ndarray) containing the probability of starting in
                     a particular hidden state.
            iterations: (int) the number of times expectation-maximization
                              should be performed.
        Returns:
            Transition, Emission, or None, None on failure.
    """
    if type(Observations) != np.ndarray or len(Observations.shape) != 1:
        return None, None
    if type(Emission) != np.ndarray or len(Emission.shape) != 2:
        return None, None
    if type(Transition) != np.ndarray or len(Transition.shape) != 2:
        return None, None
    if type(Initial) != np.ndarray or len(Initial.shape) != 2:
        return None, None
    if type(iterations) != int or iterations < 0:
        return None, None
    T = Observations.shape[0]
    N, M = Emission.shape
    if Transition.shape[0] != N or Transition.shape[0] != Transition.shape[1]:
        return None, None
    if Initial.shape[0] != N or Initial.shape[1] != 1:
        return None, None
    for i in range(iterations):
        _, A = forward(Observations, Emission, Transition, Initial)
        _, B = backward(Observations, Emission, Transition, Initial)
        tmp = np.zeros((N, N, T - 1))
        for t in range(T - 1):
            a = (np.matmul(A[:, t].T, Transition) *
                 Emission[:, Observations[t + 1]].T)
            den = np.matmul(a, B[:, t + 1])
            for j in range(N):
                num = (A[j, t] * Transition[j] *
                       Emission[:, Observations[t + 1]].T *
                       B[:, t + 1].T)
                tmp[j, :, t] = num / den
        gamma = np.sum(tmp, axis=1)
        div = np.sum(gamma, axis=1).reshape((-1, 1))

        Transition = np.sum(tmp, axis=2) / div
        summ = np.sum(tmp[:, :, T - 2], axis=0).reshape((-1, 1))
        gamma = np.hstack((gamma, summ))

        den = np.sum(gamma, axis=1).reshape((-1, 1))
        for z in range(M):
            obs = gamma[:, Observations == z]
            Emission[:, z] = np.sum(obs, axis=1)
        Emission = Emission / den

    return Transition, Emission
