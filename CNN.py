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
        self.best_accuracy = 0

    def shuffle(self):
        indices = np.arange(self.X_train.shape[0])
        np.random.shuffle(indices)
        self.X_train = self.X_train[indices]
        self.y_train = self.y_train[indices]

        val_size = self.X_train.shape[0]//8
        self.X_val = np.array(self.X_train[:val_size])
        self.y_val = np.array(self.y_train[:val_size])
        self.X_train = np.array(self.X_train[val_size:])
        self.y_train = np.array(self.y_train[val_size:])


    def predict(self, X):
        # print(X)
        for layer in self.layers:
            X = layer.forward(X)
            # print(X)
        return X

    def train(self, batch_size = 8, report = False, snapshot = True):
        
        self.best_accuracy = 0
        accuracy_train = [0 for _ in range(self.epochs)]
        accuracy_val = [0 for _ in range(self.epochs)]
        error_train = [0 for _ in range(self.epochs)]
        for epoch in range(self.epochs):

            #self.shuffle()

            error = 0
            count = 0
            if report:
                progress = tqdm(range(self.X_train.shape[0]))
                progress.set_description(f"Training epoch {epoch+1}/{self.epochs}")

            batch = 0
            batchX = np.zeros((batch_size, self.X_train.shape[1], self.X_train.shape[2], self.X_train.shape[3]))
            batchY = np.zeros((batch_size, self.y_train.shape[1], self.y_train.shape[2]))
            for x,y in zip(self.X_train, self.y_train):
                # print(x.shape, y.shape)
                if report:
                    progress.update(1)

                batchX[batch] = x
                batchY[batch] = y
                batch += 1
                
                if batch == batch_size:
                    batch = 0

                    pred = self.predict(batchX)
                    # print(pred.shape)
                    for i in range(batch_size):
                        error += self.loss.forward(batchY[i], pred[i])
                    for i in range(batch_size):
                        count += np.argmax(pred[i]) == np.argmax(batchY[i])


                    grad = [[] for _ in range(batch_size)]
                    for i in range(batch_size):
                        grad[i] = self.loss.backward(batchY[i], pred[i])
                    grad = np.array(grad)
                    for layer in reversed(self.layers):
                        grad = layer.backward(grad, self.rate)
                        # print(type(layer).__name__,grad.shape)

            val_accuracy = 0
            if self.X_val is not None:
                for x,y in zip(self.X_val, self.y_val):
                    pred = self.predict(np.array([x]))
                    val_accuracy += np.argmax(pred[0]) == np.argmax(y)
                val_accuracy /= self.X_val.shape[0]

            if report:
                progress.close()
                print(f"Error: {error/self.X_train.shape[0]}")
                print(f"Traigning accuracy: {count/self.X_train.shape[0]}")
                if self.X_val is not None:
                    print(f"Validation accuracy: {val_accuracy}")
                print("\n")
            if snapshot:
                if val_accuracy > self.best_accuracy:
                    self.best_accuracy = val_accuracy
                    self.save("best_snapshot.npy")
            
            accuracy_train[epoch] = count/self.X_train.shape[0]
            accuracy_val[epoch] = val_accuracy
            error_train[epoch] = error/self.X_train.shape[0]
            # print(accuracy_train, accuracy_val, error_train)
        return accuracy_train, accuracy_val, error_train

    
    def number_of_parameters(self):
        print([layer.number_of_parameters() for layer in self.layers])
        return sum([layer.number_of_parameters() for layer in self.layers])
    
    def stats(self):
        pass

    def reset(self):
        for layer in self.layers:
            if layer.number_of_parameters() != 0:
                layer.initialize()

    def save(self, path):
        record = np.array([])
        for layer in self.layers:
            if layer.number_of_parameters() != 0:
                record = np.append(record, layer.save())
        np.save(path, record)

    def load(self, path):
        record = np.load(path, allow_pickle=True)
        index = 0
        for layer in self.layers:
            if layer.number_of_parameters() != 0:
                layer.load(record[index: index + layer.number_of_parameters()])
                index += layer.number_of_parameters()



