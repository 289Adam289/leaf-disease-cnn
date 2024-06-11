import numpy as np



class Flatten:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = (input_shape[0]*input_shape[1]*input_shape[2], 1)

    def forward(self, input):
        batch_size = input.shape[0]

        output = np.zeros((batch_size, self.output_shape[0], self.output_shape[1]))
        for i in range(batch_size):
            output[i] = np.reshape(input[i], self.output_shape)
        return output

    def backward(self, output_gradient, learning_rate):
        batch_size = output_gradient.shape[0]
        tmp = [None for _ in range(batch_size)]
        for i in range(batch_size):
            tmp[i] = np.reshape(output_gradient[i], self.input_shape)
        return np.array(tmp)

    def number_of_parameters(self):
        return 0
    
class MaxPooling:

    def __init__(self, X_shape, kernel_size, stride):
        self.X_shape = X_shape
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, X):
        batch_size = X.shape[0]
        self.X = X

        
        channels, height, width = self.X.shape[1], self.X.shape[2], self.X.shape[3]
        new_height = (height - self.kernel_size) // self.stride + 1
        new_width = (width - self.kernel_size) // self.stride + 1
        self.output = np.zeros((batch_size,channels, new_height, new_width))

        for sample in range(batch_size):
            for c in range(channels):
                for i in range(new_height):
                    for j in range(new_width):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size
                        
                        self.output[sample, c, i, j] = np.max(X[sample, c, h_start:h_end, w_start:w_end])
        return self.output


    def backward(self, grad, rate):
        batch_size = grad.shape[0]
        dX = [np.zeros(self.X_shape) for _ in range(batch_size)]
        channels, pooled_height, pooled_width = grad.shape[1], grad.shape[2], grad.shape[3]

        for sample in range(batch_size):
            for c in range(channels):
                for i in range(pooled_height):
                    for j in range(pooled_width):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size

                        for h in range(h_start, h_end):
                            for w in range(w_start, w_end):
                                if self.X[sample, c, h, w] == grad[sample, c, i, j]:
                                    dX[sample, c, h, w] += rate
        return np.array(dX)
    
    def number_of_parameters(self):
        return 0


