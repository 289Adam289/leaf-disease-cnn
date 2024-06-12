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

# network = [
#     Convolutional((3, 128, 128), 3, 5, activation="relu"),     # 128 -> 126
#     MaxPooling((5, 126, 126), 2, 2),                          # 126 -> 63
#     Convolutional((5, 63, 63), 4, 10, activation="relu"),      # 63 -> 60
#     MaxPooling((10, 60, 60), 2, 2),                           # 60 -> 30
#     Convolutional((10, 30, 30), 3, 12, activation="relu"),      # 30 -> 28
#     MaxPooling((12, 28, 28), 2, 2),                            # 28 -> 14
#     Flatten((12,14,14)),
#     Dense(12*14*14, 50, activation="relu"),
#     Dense(50, 4, activation="softmax")
# ]

# network = [
#     Convolutional((3, 128, 128), 3, 3, activation="relu"),
#     MaxPooling((3, 126, 126), 21, 21),
#     Flatten((3, 6, 6)),
#     Dense(3*6*6, 20, activation="relu"),
#     Dense(20, 4, activation="softmax")
# ]

network = [
    Convolutional((3, 128, 128), 5, 5, activation="relu"),
    MaxPooling((5, 124, 124), 4, 4),
    Convolutional((5, 31, 31), 4, 5, activation="relu"),
    MaxPooling((5, 28, 28), 2, 2),
    Convolutional((5,14,14), 3, 10),
    MaxPooling((10,12,12),4,4),
    Flatten((10, 3, 3)),
    Dense(10*3*3, 15, activation="relu"),
    Dense(15, 4, activation="softmax")
]

soft = SoftMax()

model = CNN(network)
model.fit(x_train, y_train, None, None, Loss("crossentropy"), epochs=20, rate=0.1)
model.train(report=True)

count = 0
for x, y in zip(x_test, y_test):
    output = model.predict(x)
    # print(output)
    # print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
    count += np.argmax(output) == np.argmax(y)
print(f"Accuracy: {count / len(x_test)}")