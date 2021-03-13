import numpy as np

import sys
sys.path.insert(0, '..')
from src.utils import *

class LinearRegression:
    def __init__(self):
        self.weights = None

    def train(self, X, y, iterations=5000, learning_rate=0.01, display=False):
        '''

        Input parameters:
        X: (mxn) array where m is the number of training examples and n is the number of features
        y: (mx1) array with target values
        '''

        # We initialize weights as a (1xn) array of zeros
        self.weights = np.zeros((X.shape[1], 1))

        loss_func = MeanSquaredError()
        opt = BatchGradientDescent()
        loss_hist = np.zeros((1,0))

        for i in range(iterations):
            y_hat = X.dot(self.weights)
            loss = loss_func.loss(y, y_hat)
            loss_hist = np.append(loss_hist, loss)

            errors = loss_func.gradient(y, y_hat)
            self.weights = opt.optimize(X, errors, self.weights, learning_rate)
            if display:
                show_progress(i, epochs, loss)

        if display:
            print('\n')

        return loss_hist, loss

    def predict(self, X, y):
        loss_func = MeanSquaredError()
        y_hat = X.dot(self.weights)
        loss = loss_func.loss(y, y_hat)

        return y_hat, loss
