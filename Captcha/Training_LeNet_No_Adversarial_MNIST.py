import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from sklearn.model_selection import train_test_split
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

"""
    Here we start by opening the MNIST dataset
    We will train the model with the individual digits from MNIST
    Afterwards we will only need to separate the digits in the CAPTCHA 
"""

(trainX, trainY), (testX, testY) = mnist.load_data()

trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))

trainX = trainX.astype('float32')
testX = testX.astype('float32')

trainX = trainX / 255.0
testX = testX / 255.0

trainY = to_categorical(trainY, 10)
testY = to_categorical(testY, 10)

"""
    Here we have LeNet
    Let us train it with the MNIST dataset
"""
model = Sequential()
model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(AveragePooling2D())
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D())
model.add(Flatten())
model.add(Dense(units=120, activation='relu'))
model.add(Dense(units=84, activation='relu'))
model.add(Dense(units=10, activation = 'softmax'))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# Train the neural network
model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=10, epochs=5, verbose=1)

model.save("./MODELS/LeNet_MNIST_data.hdf5")
