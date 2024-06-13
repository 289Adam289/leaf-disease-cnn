import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers, losses, callbacks
from keras.preprocessing.image import ImageDataGenerator
import pickle

data_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    validation_split=0.2
)

data_dir = 'data'

train_generator = data_gen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = data_gen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

class_names = list(train_generator.class_indices.keys())
num_classes = len(class_names)
print(class_names)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=optimizers.Adam(lr=0.001),
    loss=losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    callbacks=[callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy')]
)

history_filename = 'history_less.pkl'
with open(history_filename, 'wb') as file:
    pickle.dump(history.history, file)

model.load_weights('best_model.h5')
model.save('model_less_convo.h5')
