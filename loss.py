import numpy as np


def mse(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))


def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)


def crossentropy(y_true, y_pred):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))


def crossentropy_derivative(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))


class Loss:
    def __init__(self, loss=["mse", "crossentropy"]):
        self.loss = loss
        if loss == "mse":
            self.func = mse
            self.dfunc = mse_derivative
        elif loss == "crossentropy":
            self.func = crossentropy
            self.dfunc = crossentropy_derivative

    def forward(self, y_true, y_pred):
        return self.func(y_true, y_pred)

    def backward(self, y_true, y_pred):
        return self.dfunc(y_true, y_pred)

    def __call__(self, y_true, y_pred):
        return self.func(y_true, y_pred)
