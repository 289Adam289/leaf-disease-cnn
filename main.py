import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from connected import Dense, Convolutional
from extra import Flatten, MaxPooling
from CNN import CNN
from loss import Loss
import data

(x_train, y_train), (x_test, y_test) = data.load_data()

print(x_train.shape)
print(y_train.shape)

network = [
    Convolutional((3, 18, 18), 3, 8, activation="relu"),
    MaxPooling((8, 16, 16), 2, 2),
    Flatten((8,8,8)),
    Dense(8*8*8, 20, activation="relu"),
    Dense(20, 2, activation="softmax")
]

model = CNN(network)

print(model.number_of_parameters())

model.fit(x_train, y_train, x_test, y_test, Loss("crossentropy"), epochs=5, rate=0.005)


#model.load("best_snapshotu.npy")
model.train(report=True, batch_size=1, snapshot=True)
model.save("leaves5model.npy")

count = 0
for x, y in zip(x_test, y_test):
    xx = np.zeros((1, 1, 28, 28))
    xx[0] = x
    output = model.predict(xx)
    # print(output)
    # print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
    count += np.argmax(output[0]) == np.argmax(y)
print(f"Accuracy: {count / len(x_test)}")
