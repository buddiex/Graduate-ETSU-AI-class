import numpy as np


class Perceptron:
    """ Perceptron classifier """

    def __init__(self, n_inputs, learning_rate=0.01, iterations=10):
        self.n_inputs = n_inputs
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w_ = np.zeros(1)
        self.errors_ = []

    def train(self, X, y):
        """
            Fits the training data to the Perceptron
        """

        # TODO: Code the Perceptron Learning Algorithm
        # (1)	Initialize weights to zero or a small random number.
        #        Don't forget w_[0] is the bias weight

        self.w_ = np.zeros(len(self.w_) + len(X[0]))
        # (2)	For the number of iterations,
        for _ in range(self.iterations):
            errors = 0
            #       (a)	For each item in data set X:
            for i in range(self.n_inputs):
                # (i)Activate the answer
                y_predicted = self.activate(np.concatenate(([1], X[i])))
                # (ii)Calculate the error based on the answer and the known y[i]
                diff = self.learning_rate * (y[i] - y_predicted)
                # (iii)Update the bias weight and input weights if the activated answer is erroneous
                self.w_[1:] += diff * X[i]
                self.w_[0] += diff
                # (b)Record the error in self.errors_
                errors += int(diff != 0.0)
                self.errors_.append(errors)

    def net_input(self, X):
        """ Calculate the Net Input """
        # TODO: Calculate the sum of the product of each input and each weight
        return sum([X[i] * self.w_[i] for i in range(len(X))])

    def activate(self, X):
        """ STEP FUNCTION: Returns the class label after the unit step """
        # TODO: Return 1 if net_input(X) >= 0.0 or -1 otherwise
        return 1 if self.net_input(X) >= 0.0 else -1
