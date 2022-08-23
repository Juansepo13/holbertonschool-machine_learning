#!/usr/bin/env python3
"""Module containing the class NeuralNetwork which defines a neural network
with one hidden layer performing binary classification."""

import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork():

    """Class that defines a neural network with one hidden layer performing
    binary classification."""

    def __init__(self, nx, nodes):
        """Innitilization function for NeuralNetwork class

        Args:
            nx (int): The number of input features
            nodes (int): The number of nodes found in the hidden layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        elif not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        elif nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__W2 = np.random.randn(1, nodes)
        self.__b1 = np.zeros((nodes, 1))
        self.__b2 = 0
        self.__A1 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter method for __W1"""
        return self.__W1

    @property
    def W2(self):
        """Getter method for __W2"""
        return self.__W2

    @property
    def b1(self):
        """Getter method for __b1"""
        return self.__b1

    @property
    def b2(self):
        """Getter method for __b2"""
        return self.__b2

    @property
    def A1(self):
        """Getter method for __A1"""
        return self.__A1

    @property
    def A2(self):
        """Getter method for __A2"""
        return self.__A2

    def forward_prop(self, X):
        """Function that calculates the forward propagation of the neural
        network.

        Args:
            X (numpy.ndarray): N-dimensional array with shape (nx, m) that
            contains the input data, where nx is the number of input features
            to the neural network and m is the number of examples.

        Returns:
            self.__A1 (numpy.ndarray[(float)]): The activated output for
                the hidden layer
            self.__A2 (float): The activated output for the neural network.
        """

        z = np.dot(self.W1, X) + self.__b1
        self.__A1 = 1/(1 + np.exp(-z))  # Sigmoid
        z = np.dot(self.W2, self.__A1) + self.__b2
        self.__A2 = 1/(1 + np.exp(-z))  # Sigmoid
        return self.__A1, self.__A2

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
        loss_array = -((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A)))
        return np.sum(loss_array) / m

    def evaluate(self, X, Y):
        """Function that valuates the neural network’s predictions.

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
        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        A = np.where(A2 >= 0.5, 1, 0)
        return A, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Function that calculates one pass of gradient descent on the neural
        network.

        Args:
            X (numpy.ndarray): N-dimensioal array with shape (nx, m) that
                contains the input data, where nx is the number of input
                features to the neuron and m is the number of examples.
            Y (numpy.ndarray): N-dimensioal array with shape (1, m) that
                contains the correct labels for the input data.
            A1 (numpy.ndarray): N-dimensioal array with shape (nodes, m) that
                contains the activated output of hidden layer for each example,
                where nodes is the number of nodes in the hidden layer and m is
                the number of examples.
            A2 (numpy.ndarray): that contains the predicted output for each
                example.
            alpha (float, optional): The learning rate. Defaults to 0.05.
        """

        m = X.shape[1]
        dZ2 = A2 - Y
        dW2 = (np.matmul(dZ2, A1.T) / m)
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dZ1 = np.multiply(np.dot(self.W2.T, dZ2), (A1 * (1 - A1)))
        dW1 = np.matmul(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m
        self.__W2 = self.W2 - (alpha * dW2)
        self.__b2 = self.b2 - (alpha * db2)
        self.__W1 = self.W1 - (alpha * dW1)
        self.__b1 = self.b1 - (alpha * db1)

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
        elif verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            elif step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        x = np.arange(0, iterations, step)
        y = []

        for i in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)
            if (i) % step == 0 or i == 0:
                cost = self.cost(Y, A2)
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
