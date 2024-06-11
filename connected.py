import numpy as np
from scipy.signal import correlate2d
from activation import Activation, SoftMax



class Dense:

    def __init__(self, input_size , output_size, activation = "relu"):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        if activation == "softmax":
            self.activator = SoftMax()
        else:
            self.activator = Activation(activation)
        self.initialize()
        # self.weights = np.zeros((output_size, input_size))
        # self.bias = np.zeros((output_size, 1))

    def forward(self, X):
        self.X = X
        tmp = np.dot(self.weights, X) + self.bias
        tmp = self.activator.forward(tmp)
        return tmp
    
    def backward(self,grad, rate):
        grad = self.activator.backward(grad, rate)
        self.weights -= rate * np.dot(grad, self.X.T)
        self.bias -= rate * grad
        return np.dot(self.weights.T, grad)
    
    def initialize(self):
        divisor = 1
        if self.activation == "relu" or "softmax":
            divisor = self.input_size
        elif self.activation == "sigmoid" or self.activation == "tanh":
            divisor = self.input_size + self.output_size
        self.weights = np.random.randn(self.output_size, self.input_size)/divisor
        self.bias = np.random.randn(self.output_size, 1)/divisor
    
    def number_of_parameters(self):
        return self.input_size * self.output_size + self.output_size
    
    def save(self):
        return np.append(self.weights.flatten(), self.bias.flatten())
    def load(self, record):
        self.weights = record[:self.input_size*self.output_size].reshape(self.output_size, self.input_size)
        self.bias = record[self.input_size*self.output_size:].reshape(self.output_size, 1)

        


class Convolutional:

    def __init__(self, X_shape, kernel_size, new_channels, activation = "relu"): 
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
        # self.kernels = np.zeros((new_channels, self.channels, kernel_size, kernel_size))
        # self.bias = np.zeros((new_channels, self.height - kernel_size + 1, self.width - kernel_size + 1))
        
        
    def forward(self, X):
        self.X = X
        self.output = np.zeros((self.new_channels, self.new_height, self.new_width))
        for nc in range(self.new_channels):
            for c in range(self.channels):
                self.output[nc] += correlate2d(X[c], self.kernels[nc, c], "valid")
            self.output[nc] += self.bias[nc]
        
        self.output = self.activator.forward(self.output)
        return self.output
    
    def backward(self, grad, rate):
        grad = self.activator.backward(grad, rate)
        grad_input = np.zeros(self.X.shape) 
        grad_kernels = np.zeros(self.kernels.shape)
        grad_bias = np.zeros(self.bias.shape)
        for nc in range(self.new_channels):
            grad_bias[nc] = np.sum(grad[nc])
            for c in range(self.channels):
                grad_kernels[nc, c] = correlate2d(self.X[c], grad[nc], "valid")
                grad_input[c] += correlate2d(grad[nc], np.rot90(self.kernels[nc, c],2), "full")
        self.update_wieghts(rate, grad_kernels, grad_bias)
        return grad_input
    
    def initialize(self):
        divisor = 1
        if self.activation == "relu":
            divisor = self.channels*self.kernel_size*self.kernel_size
        elif self.activation == "sigmoid" or self.activation == "tanh":
            divisor = self.channels*self.kernel_size*self.kernel_size + self.new_channels
        self.kernels = np.random.randn(self.new_channels,
                         self.channels,self.kernel_size, self.kernel_size)/divisor
        self.bias = np.random.randn(self.new_channels,
                        self.new_height, self.new_width)/divisor
    
    def update_wieghts(self, rate, kernel, bias):
        self.kernels -= rate * kernel
        self.bias -= rate * bias

    def number_of_parameters(self):
        return self.new_channels * self.channels * self.kernel_size * self.kernel_size \
                + self.new_channels*self.new_height*self.new_width
    
    def save(self):
        return np.append(self.kernels.flatten(), self.bias.flatten())
    
    def load(self, record):
        tmp = self.new_channels * self.channels * self.kernel_size * self.kernel_size
        self.kernels = record[:tmp].reshape(self.new_channels, self.channels, self.kernel_size, self.kernel_size)
        self.bias = record[tmp:].reshape(self.new_channels, self.new_height, self.new_width)