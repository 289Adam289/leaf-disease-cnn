import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from connected import Dense, Convolutional
from extra import Flatten, MaxPooling
from CNN import CNN
from loss import Loss


def preprocess_data(x, y, limit):
    indexes = [np.where(y == i)[0][:limit] for i in range(10)]

    all_indices = np.hstack(indexes)
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = np_utils.to_categorical(y)
    print(y.shape)
    y = y.reshape(len(y), 10, 1)
    return x, y


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 200)
x_test, y_test = preprocess_data(x_test, y_test, 100)

print(x_train.shape)
print(y_train.shape)
network = [
    Convolutional((1, 28, 28), 3, 8, activation="relu"),
    MaxPooling((8, 26, 26), 2, 2),
    Flatten((8, 13, 13)),
    Dense(8 * 13 * 13, 50, activation="relu"),
    Dense(50, 10, activation="softmax"),
]


model = CNN(network)

print(model.number_of_parameters())

model.fit(x_train, y_train, x_test, y_test, Loss("crossentropy"), epochs=55, rate=0.035)


# model.load("mnist10_2/model2.npy")
acuuracy_train, acu_val, eror = model.train(report=True, batch_size=1, snapshot=True)

np.save("mnist10_3/accuracy_train.npy", acuuracy_train)
np.save("mnist10_3/accuracy_val.npy", acu_val)
np.save("mnist10_3/error.npy", eror)


model.save("mnist10_3/model.npy")

# model.reset()

# model.load("model.npy")

count = 0
for x, y in zip(x_test, y_test):
    xx = np.zeros((1, 1, 28, 28))
    xx[0] = x
    output = model.predict(xx)
    # print(output)
    # print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
    count += np.argmax(output[0]) == np.argmax(y)
print(f"Accuracy: {count / len(x_test)}")
