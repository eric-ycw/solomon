import sys
import numpy as np

class Activation:
    def val(self, x): pass
    def deriv(self, x): pass

class Sigmoid(Activation):
    def val(self, x):
        return 1 / (1 + np.exp(-x))

    def deriv(x):
        return self.val(x) / (1 - self.val(x))

class Loss:
    @staticmethod
    def loss(y, y_hat): pass

    @staticmethod
    def gradient(X, y, y_hat): pass

class MeanSquaredError(Loss):
    @staticmethod
    def loss(y, y_hat):
        return np.square(y - y_hat).mean()

    @staticmethod
    def gradient(X, y, y_hat):
        return (-2 / X.shape[0]) * X.T.dot(y - y_hat)

class Normalization:
    def normalize(self, features): pass

class Rescale(Normalization):
    def normalize(self, features, min=-1, max=1):
        return min + (max - min) * (features - features.min(0)) / features.ptp(0)

class ZScoreNormalization(Normalization):
    @staticmethod
    def normalize(features):
        return (features - features.mean(0)) / features.std(0)

class Optimization:
    @staticmethod
    def optimize(): pass

class BatchGradientDescent(Optimization):
    @staticmethod
    def optimize(X, y, y_hat, params, learning_rate, loss_func):
        return params - loss_func.gradient(X, y, y_hat) * learning_rate

### Miscellaneous ###
def show_progress(iteration, max_iteration, loss, bar_size=40):
    prog = (iteration + 1) / max_iteration
    loss_str = 'Loss: ' + str(loss)
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(bar_size * prog):{bar_size}s}] {int(100 * prog)}%  {loss_str}")
    sys.stdout.flush()
