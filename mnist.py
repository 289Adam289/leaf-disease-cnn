import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from activation import Activation, SoftMax
from connected import Dense, Convolutional
from extra import Flatten, MaxPooling
from CNN import CNN
from loss import Loss

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

def plot_sample_images(x, y):
    samples = {}

    for i in range(len(x)):
        label = np.argmax(y[i])
        if label not in samples:
            samples[label] = x[i]
            if len(samples) == 10:
                break

    plt.figure(figsize=(10, 2))
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.imshow(samples[i].reshape(28, 28), cmap='gray')
        plt.title(f"Class {i}")
        plt.axis('off')
    plt.show()

def preprocess_data(x, y):
    indices = np.random.permutation(len(x))
    x, y = x[indices], y[indices]

    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255

    y = to_categorical(y, num_classes=10) 
    y = np.array([label[:, np.newaxis] for label in y])
    
    return x, y

(x_train, y_train), (x_test, y_test) = mnist.load_data()

set_size = 100
x_train, y_train = preprocess_data(x_train[:set_size], y_train[:set_size])
x_test, y_test = preprocess_data(x_test[:set_size], y_test[:set_size])

print(x_train.shape)
print(y_train.shape)

plot_sample_images(x_train, y_train)

network = [
    Convolutional((1, 28, 28), 3, 32, activation="relu"),
    MaxPooling((32, 26, 26), 2, 2),
    Flatten((32,13,13)),
    Dense(32*13*13, 50, activation="relu"),
    Dense(50, 10, activation="softmax")
]

# network = [
#     Convolutional((1, 28, 28), 3, 8, activation="relu"),
#     MaxPooling((8, 26, 26), 2, 2),
#     Convolutional((8, 13,13), 4, 16),
#     MaxPooling((16,10,10), 2, 2),
#     Flatten((16,5,5)),
#     Dense(16*5*5, 50, activation="relu"),
#     Dense(50, 10, activation="softmax")
# ]
soft = SoftMax()

model = CNN(network)
model.fit(x_train, y_train, None, None, Loss("crossentropy"), epochs=20, rate=0.1)
model.train(report=True)

count = 0
for x, y in zip(x_test, y_test):
    output = model.predict(x)
    count += np.argmax(output) == np.argmax(y)
print(f"Accuracy: {count / len(x_test)}")