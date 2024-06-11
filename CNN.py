import numpy as np

from tqdm import tqdm
import time

class CNN:

    def __init__(self, layers: np.array):
        self.layers = layers

    
    def fit(self, X_train, y_train, X_val, y_val, loss, epochs=100, rate = 0.01):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.epochs = epochs
        self.loss = loss
        self.rate = rate

    def predict(self, X):
        # print(X)
        for layer in self.layers:
            X = layer.forward(X)
            # print(X)
        return X

    def train(self, report = False):
        
        for epoch in range(self.epochs):
            error=  0
            count = 0
            if report:
                progress = tqdm(range(self.X_train.shape[0]))
                progress.set_description(f"Training epoch {epoch+1}/{self.epochs}")


            for x,y in zip(self.X_train, self.y_train):
                if report:
                    progress.update(1)

                pred = self.predict(x)
                error += self.loss.forward(y, pred)
                count += np.argmax(pred) == np.argmax(y)
                grad = self.loss.backward(y, pred)
                # print(f'grad {grad}')
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, self.rate)
                    # print(f'grad {grad}')

            if report:
                progress.close()
                print(f"Error: {error/self.X_train.shape[0]}")
                print(f"Traigning accuracy: {count/self.X_train.shape[0]}")
                print("\n")
    #TODO: Implement save and load methods
    def save(self, path):
        pass

    def load(self, path):
        pass



