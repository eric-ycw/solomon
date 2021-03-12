import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '..')
from src.regression.linear_regression import LinearRegression
from src.utils import *

x = 30 * np.random.random((20, 1))
y = 0.5 * x + 1.0 + np.random.normal(size=x.shape)

x, y = mean_norm(x), mean_norm(y)

lr = LinearRegression()
params, cost = lr.simple_linear_regression(x, y, display=True)

ax = plt.axes()
ax.scatter(x, y)
x_new = np.linspace(-3, 3, 100)
ax.plot(x_new, params[0] * x_new + params[1])

ax.set_xlabel('x')
ax.set_ylabel('y')

ax.axis('tight')

plt.show()
