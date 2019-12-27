import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
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

print(trainX[0])
exit()

trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))

trainX = trainX.astype('float32')
testX = testX.astype('float32')

trainX = trainX / 255.0
testX = testX / 255.0

trainY = to_categorical(trainY, 10)
testY = to_categorical(testY, 10)

"""
    Here we have the model
    Train a deep convolutional network to recognize the digits
"""
# Build the neural network!
model = Sequential()
# First convolutional layer with max pooling
model.add(Conv2D(20, (5, 5), padding="same", input_shape=(28, 28, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# Second convolutional layer with max pooling
model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# Hidden layer with 500 nodes
model.add(Flatten())
model.add(Dense(500, activation="relu"))
# Output layer with 32 nodes (one for each possible letter/number we predict)
model.add(Dense(10, activation="softmax"))
# Ask Keras to build the TensorFlow model behind the scenes
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# Train the neural network
model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=10, epochs=5, verbose=1)

model.save("./MODELS/captcha_model.hdf5")
