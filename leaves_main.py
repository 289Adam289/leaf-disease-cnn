import sys
from matplotlib import pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from connected import Dense, Convolutional
from extra import Flatten, MaxPooling
from CNN import CNN
from loss import Loss
import data


def make_model():

    (x_train, y_train), (x_test, y_test) = data.load_data()

    print(x_train.shape)
    print(y_train.shape)

    network = [
        Convolutional((3, 18, 18), 3, 8, activation="relu"),
        MaxPooling((8, 16, 16), 2, 2),
        Flatten((8, 8, 8)),
        Dense(8 * 8 * 8, 20, activation="relu"),
        Dense(20, y_train.shape[1], activation="softmax"),
    ]

    model = CNN(network)

    model.fit(
        x_train, y_train, x_test, y_test, Loss("crossentropy"), epochs=60, rate=0.005
    )
    model.load("leaves_models/best_snapshot.npy")
    model.train(report=True, batch_size=1, snapshot=True)
    #model.save("leaves_models/leaves5model.npy")

    count = 0
    for x, y in zip(x_test, y_test):
        xx = np.zeros((1, 3, 18, 18))
        xx[0] = x
        output = model.predict(xx)
        count += np.argmax(output[0]) == np.argmax(y)
    print(f"Accuracy: {count / len(x_test)}")


def classify_leaf():

    classes = ["healthy", "blue", "rot", "scab", "rust"]

    network = [
        Convolutional((3, 18, 18), 3, 8, activation="relu"),
        MaxPooling((8, 16, 16), 2, 2),
        Flatten((8, 8, 8)),
        Dense(8 * 8 * 8, 20, activation="relu"),
        Dense(20, 5, activation="softmax"),
    ]

    model = CNN(network)
    model.load("leaves_models/good_model.npy")
    # model.load("models/not_trained_model.npy")

    leaf_no = int(sys.argv[1])

    for i in range(5):

        photo_path = f"../nice_leaves/final_{classes[i]}/leaf{leaf_no}.jpg"
        leaf = data.get_leaf(photo_path)
        prepared = np.zeros((1, 3, 18, 18))
        prepared[0] = leaf
        output = model.predict(prepared)
        prediction = np.argmax(output[0])
        print(output)
        print("prediction: ", prediction)

        flattened_data = output.flatten()

        image_array = leaf.transpose(1, 2, 0)

        plt.rcParams.update(
            {"font.size": 18, "font.family": "serif", "font.serif": "Times New Roman"}
        )

        fig, axs = plt.subplots(
            2, 1, figsize=(5, 10), gridspec_kw={"height_ratios": [3, 2]}
        )

        axs[1].bar(range(len(flattened_data)), flattened_data, color="skyblue")
        axs[1].set_xticks(range(len(flattened_data)))
        axs[1].set_xticklabels(classes)
        axs[1].set_yticks([])

        axs[0].imshow(image_array)
        axs[0].set_title(f"{classes[i]}")
        axs[0].axis("off")

        plt.tight_layout()
        plt.show()


# make_model()

classify_leaf()
