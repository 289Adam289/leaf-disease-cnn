import sys
import numpy as np
import tensorflow as tf
from keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt


def load_image(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img)
    print(img_array.shape)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


def predict(image_path, model, class_names):
    img_array = load_image(image_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    return class_names[predicted_class[0]]

if __name__ == "__main__":

    image_path = sys.argv[1]
    class_names = ['healthy', 'rust', 'scab']

    print("haha")

    model = tf.keras.models.load_model('simple_cnn_leaf_classifier2.h5')

    print("haha")

    prediction = predict(image_path, model, class_names)
    print(f'Predicted class: {prediction}')
    img = load_img(image_path)
    plt.imshow(img)
    plt.title(f'Predicted: {prediction}')
    plt.axis('off')
    plt.show()
