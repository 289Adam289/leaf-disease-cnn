import numpy as np
from activation import SoftMax
from connected import Dense, Convolutional
from extra import Flatten, MaxPooling
from CNN import CNN
from loss import Loss
import data

(x_train, y_train), (x_test, y_test) = data.load_data()

print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")


network = [
    Convolutional((3, 16, 16),3, 8, activation="relu"),
    MaxPooling((8, 14, 14), 2, 2),
    Flatten((8, 7, 7)),
    Dense(8*7*7, 30, activation="relu"),
    Dense(30, 2, activation="softmax")
]

# network = [
#     Convolutional((3, 128, 128), 5, 5, activation="relu"),
#     MaxPooling((5, 124, 124), 4, 4),
#     Convolutional((5, 31, 31), 4, 5, activation="relu"),
#     MaxPooling((5, 28, 28), 2, 2),
#     Convolutional((5,14,14), 3, 10),
#     MaxPooling((10,12,12),4,4),
#     Flatten((10, 3, 3)),
#     Dense(10*3*3, 15, activation="relu"),
#     Dense(15, 4, activation="softmax")
# ]

soft = SoftMax()

model = CNN(network)
model.fit(x_train, y_train, None, None, Loss("crossentropy"), epochs=30, rate=0.001)
model.train(report=True,batch_size=8)

count = 0
for x, y in zip(x_test, y_test):
    output = model.predict(x)
    # print(output)
    # print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
    count += np.argmax(output) == np.argmax(y)
print(f"Accuracy: {count / len(x_test)}")