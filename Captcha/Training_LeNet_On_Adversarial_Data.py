import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from sklearn.model_selection import train_test_split
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
import cv2
import os

"""
    Here we need to use the own Captchas to extract our training data
    So we start by opening the ADVERSARIAL_DATABASE
"""

path = "./FILTER_ADVERSARIAL_DATABASE/"

trainX = []
trainY = []

testX = []
testY = []

"""
    Here we extract each image from the Captcha 
    And using the label in the title of the image we train the dataset!
    K is used to limit the number of digits we use to train
"""
listing = os.listdir(path)
K = 0
flag_validation_data = 0
for image_name in listing:

    # K limits the number of digits, break when to high
    if K >= 10000:
        flag_validation_data = 1
    elif K >= 15000:
        break
    K = K + 1

    # Read the image
    # Convert to grayscale and the proper way for the model input
    img = cv2.imread(path + image_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = 255 - img

    # Now we divide the digits to train the model digit by digit!
    # Since the digits are 28 pixels apart from each other
    # We use this information to divide the digits
    x = 0
    while x != 112:

        letter_image = img[0:28, x:x + 28]
        x = x + 28

        if flag_validation_data == 0:
            trainX.append(letter_image)
        else:
            testX.append(letter_image)

    # Here we take the digits solution by using the filename!
    # The four last digits before the ext are the digits we are looking for
    flag = 0
    for letter in image_name:
        if letter == ".":
            break
        if letter == "_":
            flag = 1
            continue
        if flag == 0:
            continue

        if flag_validation_data == 0:
            trainY.append(int(letter))
        else:
            testY.append(int(letter))

trainX = np.asarray(trainX)
testX = np.asarray(testX)

"""
    Now with your test and train data ready we can prepare our deep learning model
    We use exactly the same model as the last time
    Let's see how this goes!
"""

# First prepare the data for the deep learning model!!!!!
# Set the shape and normalize the data
# Also set the labels to categorical
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


model.save("./MODELS/LeNet_Adversarial_Training.hdf5")




