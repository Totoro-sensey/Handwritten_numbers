from config import MODEL, TRAIN_IMAGE
import cv2
import numpy as np
from tensorflow.keras.datasets import mnist         # библиотека базы выборок Mnist
from tensorflow import keras
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

# Model 1 keras: 3 hidden cnn
model = keras.Sequential([
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2), strides=2),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(10, activation='softmax')
])


def model_train():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype(np.float32)
    x_train /= 255.0
    x_test = x_test.astype(np.float32)
    x_test /= 255.0

    y_train_cat = keras.utils.to_categorical(y_train, 10)
    y_test_cat = keras.utils.to_categorical(y_test, 10)

    cv2.imwrite(TRAIN_IMAGE, x_train[0])

    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)

    learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5,
                                                                min_lr=0.00001)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train_cat, batch_size=32, epochs=7, validation_split=0.2, callbacks=[learning_rate_reduction])
    model.evaluate(x_test, y_test_cat)
    model.save(MODEL)

