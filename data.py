import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


def load_images_from_folder(folder, image_size=(18, 18)):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            with Image.open(img_path) as img:
                img = img.resize(image_size)
                img_array = np.array(img)
                if img_array.shape == (image_size[0], image_size[1], 3):
                    img_array = np.transpose(img_array, (2, 0, 1))
                    images.append(img_array)
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
    return np.array(images)


data_dir = "./../nice_leaves"


def get_leaf(path, image_size=(18, 18)):
    try:
        with Image.open(path) as img:
            img = img.resize(image_size)
            img_array = np.array(img)
            img_array = np.transpose(img_array, (2, 0, 1))
    except Exception as e:
        print("Error lading image")
    return img_array


def process_data():

    class_folders = [
        os.path.join(data_dir, folder)
        for folder in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, folder))
    ]

    class_images = []
    for folder in class_folders:
        images = load_images_from_folder(folder)
        class_images.append(images)

    x_train_list = []
    y_train_list = []

    x_test_list = []
    y_test_list = []

    classes_no = len(class_folders)

    for class_idx, images in enumerate(class_images):
        if len(images) > 0:
            x_train, x_test = train_test_split(images, test_size=0.3, random_state=42)
            x_train_list.extend(x_train)
            y_train_list.extend(
                [np.eye(classes_no)[class_idx][:, np.newaxis]] * len(x_train)
            )
            x_test_list.extend(x_test)
            y_test_list.extend(
                [np.eye(classes_no)[class_idx][:, np.newaxis]] * len(x_test)
            )

    x_train = np.array(x_train_list)
    y_train = np.array(y_train_list)

    x_test = np.array(x_test_list)
    y_test = np.array(y_test_list)

    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

    print(x_train.shape)
    print(y_train.shape)

    np.savez(
        "./../nice_leaves.npz",
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )


def load_data():
    data = np.load("./../nice_leaves.npz")
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]
    return (x_train, y_train), (x_test, y_test)


process_data()
