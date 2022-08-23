#!/usr/bin/env python3
"""Module containing the class DeepNeuralNetwork which defines a deep neural
network performing binary classification."""

import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork():
    """Class that defines a deep neural network performing binary
    classification."""

    def __init__(self, nx, layers, activation='sig'):
        """Class constructor

        Args:
            nx (int): The number of input features
            layers (list[int]): List representing the number of nodes in each
                layer of the network
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        elif not isinstance(layers, list) or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")
        elif activation is not 'sig' and activation is not 'tanh':
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__activation = activation
        self.__cache = {}
        weights = {}
        prev = nx
        for i, L in enumerate(layers, 1):
            if not isinstance(L, int) and L < 0:
                raise TypeError("layers must be a list of positive integers")
            weights["b{}".format(i)] = np.zeros((L, 1))
            weights["W{}".format(i)] = (np.random.randn(L, prev) *
                                        np.sqrt(2 / prev))
            prev = L
        self.__weights = weights

    @property
    def L(self):
        """Getter for self.__L"""
        return self.__L

    @property
    def cache(self):
        """Getter for self.__cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for self.__weights"""
        return self.__weights

    @property
    def activation(self):
        """Getter for self.__activation"""
        return self.__activation

    def forward_prop(self, X):
        """Function that calculates the forward propagation of the deep
            neural network.

        Args:
            X (numpy.ndarray): N-dimensional array with shape (nx, m) that
            contains the input data, where nx is the number of input features
            to the deep neural network and m is the number of examples.

        Returns:
            A (numpy.ndarray[(float)]): The activated output for the deep
            neural network.
        """
        A = X
        self.cache["A{}".format(0)] = X
        for L in range(1, self.L + 1):
            prev = A
            W = self.weights["W{}".format(L)]
            b = self.weights["b{}".format(L)]
            Z = np.matmul(W, A) + b
            if L == self.L:
                A = self.softmax(Z)
            else:
                if self.activation == 'sig':
                    A = self.sigmoid(Z)
                if self.activation == 'tanh':
                    A = self.tanh(Z)
            self.__cache["A{}".format(L)] = A
        return A, self.cache

    def cost(self, Y, A):
        """Function that alculates the cost of the model using
            logistic regression.

        Args:
            Y (numpy.ndarray): N-dimensional array with shape (1, m) that
                contains the correct labels for the input data.
            A (numpy.ndarray): N-dimensioal array with shape (1, m) containing
                the activated output of the neuron for each example.
                Sometiems refered to as "y hat" a y with a "^" above it.

        Returns:
            The cost of the model.
        """
        m = Y.shape[1]
        loss = np.sum(-Y * np.log(A))
        return loss / m

    def evaluate(self, X, Y):
        """Function that valuates the deep neural network’s predictions.

        Args:
            X (numpy.ndarray): N-dimensioal array with shape (nx, m) that
                contains the input data, where nx is the number of input
                features to the neuron and m is the number of examples.
            Y (numpy.ndarray): N-dimensioal array with shape (1, m) that
                contains the correct labels for the input data.

        Returns:
            A (numpy.ndarray): The neuron’s prediction. The predictions shape
                will be (1, m), containing the predicted labels for each
                example.
            cost (float): The cost of the network.
        """
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        A = np.where(A >= 0.5, 1, 0)
        return A, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Function that calculates one pass of gradient descent on the neural
        network.

        Args:
            Y (numpy.ndarray): N-dimensioal array with shape (1, m) that
                contains the correct labels for the input data.
            cache (dictionary): Dictionary containing all the intermediary
                values of the network.
            alpha (float, optional): The learning rate. Defaults to 0.05.
        """
        m = Y.shape[1]

        for i in range(self.L, 0, -1):
            Ai = cache["A{}".format(i)]
            Ai_next = cache["A{}".format(i - 1)]

            if i == self.L:
                dZi = (Ai - Y)
            else:
                if self.activation == 'sig':
                    dZi = dAi_next * self.sigmoid_prime(Ai)
                elif self.activation == 'tanh':
                    dZi = dAi_next * self.tanh_prime(Ai)
            dWi = np.matmul(dZi, Ai_next.T) / m
            dbi = np.sum(dZi, axis=1, keepdims=True) / m

            dZ_prev = dZi
            Wi = self.weights["W{}".format(i)]
            dAi_next = np.matmul(Wi.T, dZi)

            self.__weights["W{}".format(i)] -= (alpha * dWi)
            self.__weights["b{}".format(i)] -= (alpha * dbi)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Function that trains the neural network.

        Args:
            X (numpy.ndarray): N-dimensioal array with shape (nx, m) that
                contains the input data, where nx is the number of input
                features to the neuron and m is the number of examples.
            Y (numpy.ndarray): N-dimensioal array with shape (1, m) that
                contains the correct labels for the input data.
            iterations (int, optional): The number of iterations to train over.
                Defaults to 5000.
            alpha (float, optional): The learning rate. Defaults to 0.05.
            verbose (boolean, optional): Defines whether or not to print
                information about the training. If True, prints "Cost after
                {iteration} iterations: {cost}" every step iterations. Includes
                data from the 0th and last iteration.
            step (int, optoinal): The number of iterations between printing
                using the verbose.
            graph (boolean, optional): Defines whether or not to graph
                information about the training once the training has completed.
                If True, plots the training data every step iterations.

        Returns:
            Evaluation of the training data after iterations of training
            have occurred.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        elif iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        elif not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        elif alpha <= 0:
            raise ValueError("alpha must be positive")

        x = np.arange(0, iterations, step)
        y = []

        for i in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            if (i) % step == 0 or i == 0:
                cost = self.cost(Y, A)
                y.append(cost)
                if verbose:
                    print("Cost after {} iterations: {}"
                          .format((i), cost))
        if verbose:
            print("Cost after {} iterations: {}"
                  .format((i + 1), cost))

        if graph:
            plt.plot(x, y, "b-")
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Function that saves the instance object to a file in pickle format.
        If filename does not have the extension .pkl, it will be added.

        Args:
            filename (str): The file to which the object will be saved.
        """
        ext = ".pkl"
        if ext not in filename:
            filename += ext
        try:
            with open(filename, "wb") as file:
                pickle.dump(self, file)
        except Exception:
            return None

    def load(filename):
        """Function that loads a pickled DeepNeuralNetwork object.

        Args:
            filename ([type]): The file from which the object should be loaded

        Returns:
            The loaded object, or None if filename doesn’t exist.
        """
        try:
            with open(filename, "rb") as file:
                contents = pickle.load(file)
                return contents
        except Exception:
            return None

    def sigmoid(self, Z):
        """Does the math for the Sigmoid activation function.

        Args:
            Z (numpy.ndarray): N-demensional array of the dZ values of each
        node in a neural network.

        Returns:
            Activation values of the layer of nodes.
        """
        return 1/(1 + np.exp(-Z))

    def sigmoid_prime(self, A):
        """Does the math for the Sigmoid` activation function.

        Args:
            A (numpy.ndarray): N-demensional array of the activation values
            of the layer of nodes.

        Returns:
            dZ values of each node in a neural network.
        """
        return (A * (1 - A))

    def softmax(self, Z):
        """Does the math for the Softmax activation function.

        Args:
            Z (numpy.ndarray): N-demensional array of the dZ values of each
        node in a neural network.

        Returns:
            Activation values of the layer of nodes.
        """
        T = np.exp(Z)
        return T / np.sum(T, axis=0, keepdims=True)

    def tanh(self, Z):
        """Does the math for the Tanh activation function.

        Args:
            Z (numpy.ndarray): N-demensional array of the dZ values of each
        node in a neural network.

        Returns:
            Activation values of the layer of nodes.
        """
        return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

    def tanh_prime(self, A):
        """Does the math for the Tanh` activation function.

        Args:
            A (numpy.ndarray): N-demensional array of the activation values
            of the layer of nodes.

        Returns:
            dZ values of each node in a neural network.
        """
        return 1 - (A ** 2)
