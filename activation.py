import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    x = sigmoid(x)
    return x * (1 - x)

def relu(x):
    return np.where(x > 0, x, x * 0.01)

def relu_derivative(x):
    return np.where(x > 0, 1, 0.01)

def Tanh(x):
    return np.tanh(x)

def Tanh_derivative(x):
    return 1 - np.tanh(x)**2


class Activation:
    
    def __init__(self, activation = ["sigmoid", "relu", "tanh"]):
        self.activation = activation
        if activation == "sigmoid":
            self.func = sigmoid
            self.dfunc = sigmoid_derivative
        elif activation == "relu":
            self.func = relu
            self.dfunc = relu_derivative
        elif activation == "tanh":
            self.func = Tanh
            self.dfunc = Tanh_derivative

    
    def forward(self, X):
        self.X =X
        return self.func(X)
    
    def backward(self, grad, rate):
        return np.multiply(grad, self.dfunc(self.X))
    

class SoftMax:

    def __init__(self):
        pass
    def forward(self, X):
        tmp = np.exp(X)
        self.output = tmp / np.sum(tmp)
        return self.output
    
    def backward(self, output_gradient, rate):
        n = np.size(self.output)
        return np.dot(rate*(np.identity(n) - self.output.T) * self.output, output_gradient)