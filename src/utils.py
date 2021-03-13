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
    def loss(self, y, y_hat): pass
    def gradient(self, y, y_hat): pass

class MeanSquaredError(Loss):
    def loss(self, y, y_hat):
        return np.square(y - y_hat).mean()

    def gradient(self, y, y_hat):
        return -(y - y_hat)

class Normalization:
    def normalize(self, features): pass

class Rescale(Normalization):
    def normalize(self, features, min=-1, max=1):
        return min + (max - min) * (features - features.min(0)) / features.ptp(0)

class MeanNormalization(Normalization):
    def normalize(self, features):
        return (features - features.mean(0)) / features.std(0)

class Optimization:
    def optimize(self, features): pass

class BatchGradientDescent(Optimization):
    def optimize(self, X, errors, weights, learning_rate):
        delta = X.T.dot(errors) / X.shape[0]

        return weights - delta * learning_rate

### Miscellaneous ###
def show_progress(iteration, max_iteration, loss, bar_size=40):
    prog = (iteration + 1) / max_iteration
    loss_str = 'Loss: ' + str(loss)
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(bar_size * prog):{bar_size}s}] {int(100 * prog)}%  {loss_str}")
    sys.stdout.flush()
