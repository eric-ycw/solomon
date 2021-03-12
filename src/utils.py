import sys
import numpy as np

### Cost functions ###
def mse(y, y_hat):
    # Mean squared error
    return np.square(y - y_hat).mean()

### Feature scaling ###
def minmax_norm(features, min=-1, max=1):
    # Minmax normalization
    return min + (max - min) * (features - features.min(0)) / features.ptp(0)

def mean_norm(features):
    # Mean normalization
    return (features - features.mean(0)) / features.std(0)

### Optimization ###
def gradient_descent(gradient, features, step):
    # Simplest form of gradient descent
    return features - gradient * step

### Miscellaneous ###
def show_progress(epoch, max_epoch, cost, bar_size=40):
    prog = (epoch + 1) / max_epoch
    cost_str = 'Cost: ' + str(cost)
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(bar_size * prog):{bar_size}s}] {int(100 * prog)}%  {cost_str}")
    sys.stdout.flush()
