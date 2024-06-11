import numpy as np



class Flatten:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = (input_shape[0]*input_shape[1]*input_shape[2], 1)

    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)
    
class MaxPooling:

    def __init__(self, X_shape, kernel_size, stride):
        self.X_shape = X_shape
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, X):
        self.X = X
        channels, height, width = self.X_shape
        new_height = (height - self.kernel_size) // self.stride + 1
        new_width = (width - self.kernel_size) // self.stride + 1
        self.output = np.zeros((channels, new_height, new_width))
        for c in range(channels):
            for i in range(new_height):
                for j in range(new_width):
                    h_start = i * self.stride
                    h_end = h_start + self.kernel_size
                    w_start = j * self.stride
                    w_end = w_start + self.kernel_size
                    
                    self.output[c, i, j] = np.max(X[c, h_start:h_end, w_start:w_end])
        return self.output



    def backward(self, grad, rate):
        dX = np.zeros_like(self.X)
        channels, pooled_height, pooled_width = grad.shape
        for c in range(channels):
            for i in range(pooled_height):
                for j in range(pooled_width):
                    h_start = i * self.stride
                    h_end = h_start + self.kernel_size
                    w_start = j * self.stride
                    w_end = w_start + self.kernel_size

                    for h in range(h_start, h_end):
                        for w in range(w_start, w_end):
                            if self.X[c, h, w] == self.output[c,i,j]:
                                dX[c, h, w] += rate*grad[c, i, j]
        
        return dX


