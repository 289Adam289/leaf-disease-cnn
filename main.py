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
    Convolutional((3, 128, 128), 3, 5, activation="relu"),     # 128 -> 126
    MaxPooling((5, 126, 126), 2, 2),                          # 126 -> 63
    Convolutional((5, 63, 63), 4, 10, activation="relu"),      # 63 -> 60
    MaxPooling((10, 60, 60), 2, 2),                           # 60 -> 30
    Convolutional((10, 30, 30), 3, 12, activation="relu"),      # 30 -> 28
    MaxPooling((12, 28, 28), 2, 2),                            # 28 -> 14
    Flatten((12,14,14)),
    Dense(12*14*14, 50, activation="relu"),
    Dense(50, 4, activation="softmax")
]

# network = [
#     Convolutional((3, 128, 128), 3, 5, activation="relu"),
#     MaxPooling((5, 126, 126), 2, 2),
#     Convolutional((5, 63, 63), 4, 8, activation="relu"),
#     MaxPooling((8, 60, 60), 2, 2),
#     Flatten((8, 30, 30)),
#     Dense(8*30*30, 40, activation="relu"),
#     Dense(40, 4, activation="softmax")
# ]

soft = SoftMax()

model = CNN(network)
#model.load('little_epoch_0_1.npy')
model.fit(x_train, y_train, None, None, Loss("crossentropy"), epochs=20, rate=0.1)
model.train(report=True)

count = 0
for x, y in zip(x_test, y_test):
    output = model.predict(x)
    # print(output)
    # print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
    count += np.argmax(output) == np.argmax(y)
print(f"Accuracy: {count / len(x_test)}")