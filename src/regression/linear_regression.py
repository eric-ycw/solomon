import numpy as np

import sys
sys.path.insert(0, '..')
from src.utils import *

class LinearRegression:

    def simple_linear_regression(self, x, y, m_hat=0, b_hat=0, epochs=5000, step=0.001, display=False):
        for i in range(epochs):
            y_hat = m_hat * x + b_hat
            cost = mse(y, y_hat)
            if display:
                show_progress(i, epochs, cost)
            params = gradient_descent(
            np.array([
                -2 * np.multiply(x, y - y_hat).mean(),
                -2 * (y - y_hat).mean()
            ]),
            np.array([m_hat, b_hat]), step)
            m_hat, b_hat = params[0], params[1]
        return params, cost
