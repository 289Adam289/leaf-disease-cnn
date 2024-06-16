import numpy as np
from scipy.signal import correlate2d
from activation import Activation, SoftMax


class Dense:

    def __init__(self, input_size, output_size, activation="relu"):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        if activation == "softmax":
            self.activator = SoftMax()
        else:
            self.activator = Activation(activation)
        self.initialize()

    def forward(self, X):
        batch_size = X.shape[0]
        self.X = X
        tmp = np.zeros((batch_size, self.output_size, 1))
        for i in range(batch_size):
            tmp[i] = np.dot(self.weights, X[i]) + self.bias
            tmp[i] = self.activator.forward(tmp[i])
        return tmp

    def backward(self, grad, rate):
        batch_size = grad.shape[0]
        tmp = [None for _ in range(batch_size)]
        for i in range(batch_size):
            tmp[i] = grad[i]
            tmp[i] = self.activator.backward(tmp[i], rate)
            self.weights -= rate * np.dot(tmp[i], self.X[i].T)
            self.bias -= rate * tmp[i]
            tmp[i] = np.dot(self.weights.T, tmp[i])
        tmp = np.array(tmp)

        return tmp

    def initialize(self):
        divisor = 1
        if self.activation == "relu" or "softmax":
            divisor = self.input_size
        elif self.activation == "sigmoid" or self.activation == "tanh":
            divisor = self.input_size + self.output_size
        self.weights = np.random.randn(self.output_size, self.input_size) / divisor
        self.bias = np.random.randn(self.output_size, 1) / divisor

    def number_of_parameters(self):
        return self.input_size * self.output_size + self.output_size

    def save(self):
        return np.append(self.weights.flatten(), self.bias.flatten())

    def load(self, record):
        self.weights = record[: self.input_size * self.output_size].reshape(
            self.output_size, self.input_size
        )
        self.bias = record[self.input_size * self.output_size :].reshape(
            self.output_size, 1
        )


class Convolutional:

    def __init__(self, X_shape, kernel_size, new_channels, activation="relu"):
        self.channels, self.height, self.width = X_shape
        self.kernel_size = kernel_size
        self.new_channels = new_channels
        self.new_height = self.height - kernel_size + 1
        self.new_width = self.width - kernel_size + 1
        self.activation = activation
        if activation == "softmax":
            self.activator = SoftMax()
        else:
            self.activator = Activation(activation)
        self.initialize()

    def forward(self, X):
        batch_size = X.shape[0]
        self.X = X
        self.output = np.zeros(
            (batch_size, self.new_channels, self.new_height, self.new_width)
        )
        for i in range(batch_size):
            for nc in range(self.new_channels):
                for c in range(self.channels):
                    self.output[i, nc] += correlate2d(
                        X[i, c], self.kernels[nc, c], "valid"
                    )
                self.output[i, nc] += self.bias[nc]
            self.output[i] = self.activator.forward(self.output[i])
        return self.output

    def backward(self, grad, rate):
        batch_size = grad.shape[0]

        grad_input = np.zeros(self.X.shape)
        grad_kernels = np.zeros((batch_size,) + self.kernels.shape)
        grad_bias = np.zeros((batch_size,) + self.bias.shape)

        for i in range(batch_size):
            for nc in range(self.new_channels):
                grad_bias[i, nc] = np.sum(grad[i, nc])
                for c in range(self.channels):
                    grad_kernels[i, nc, c] = correlate2d(
                        self.X[i, c], grad[i, nc], "valid"
                    )
                    grad_input[i, c] += correlate2d(
                        grad[i, nc], np.rot90(self.kernels[nc, c], 2), "full"
                    )
            self.update_wieghts(rate, grad_kernels[i], grad_bias[i])
        return grad_input

    def initialize(self):
        divisor = 1
        if self.activation == "relu":
            divisor = self.channels * self.kernel_size * self.kernel_size
        elif self.activation == "sigmoid" or self.activation == "tanh":
            divisor = (
                self.channels * self.kernel_size * self.kernel_size + self.new_channels
            )
        self.kernels = (
            np.random.randn(
                self.new_channels, self.channels, self.kernel_size, self.kernel_size
            )
            / divisor
        )
        self.bias = (
            np.random.randn(self.new_channels, self.new_height, self.new_width)
            / divisor
        )

    def update_wieghts(self, rate, kernel, bias):
        self.kernels -= rate * kernel
        self.bias -= rate * bias

    def number_of_parameters(self):
        return (
            self.new_channels * self.channels * self.kernel_size * self.kernel_size
            + self.new_channels * self.new_height * self.new_width
        )

    def save(self):
        return np.append(self.kernels.flatten(), self.bias.flatten())

    def load(self, record):
        tmp = self.new_channels * self.channels * self.kernel_size * self.kernel_size
        self.kernels = record[:tmp].reshape(
            self.new_channels, self.channels, self.kernel_size, self.kernel_size
        )
        self.bias = record[tmp:].reshape(
            self.new_channels, self.new_height, self.new_width
        )
