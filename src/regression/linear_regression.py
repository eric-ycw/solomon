import numpy as np

import sys
sys.path.insert(0, '..')
from src.utils import *

class LinearRegression:
    def __init__(self):
        self.params = None

    def train(self, X, y, iterations=5000, learning_rate=0.01, display=False):
        '''

        Input parameters:
        X: (mxn) array where m is the number of training examples and n is the number of features
        y: (mx1) array with target values
        '''

        # We initialize parameters as a (1xn) array of zeros
        self.params = np.zeros((X.shape[1], 1))

        loss_hist = np.zeros((1,0))

        for i in range(iterations):
            y_hat = X.dot(self.params)
            loss = MeanSquaredError.loss(y, y_hat)
            loss_hist = np.append(loss_hist, loss)

            self.params = BatchGradientDescent.optimize(
                X, y, y_hat, self.params, learning_rate, MeanSquaredError)
            if display:
                show_progress(i, iterations, loss)

        if display:
            print('\n')

        return loss_hist, loss

    def predict(self, X, y):
        y_hat = X.dot(self.params)
        loss = MeanSquaredError.loss(y, y_hat)

        return y_hat, loss
