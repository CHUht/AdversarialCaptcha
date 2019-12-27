from sklearn.model_selection import train_test_split
import pandas as pd

import numpy as np
import keras
from keras import backend
from keras.models import load_model
from keras.datasets import mnist
import tensorflow as tf
import os
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.utils_keras import KerasModelWrapper
from keras.utils import to_categorical

from matplotlib import pyplot as plt
import imageio

# Set the learning phase to false, the model is pre-trained.
backend.set_learning_phase(False)
path = os.getcwd()
path = path.replace("\TEST","")
path = path + "/MODELS/LeNet_MNIST_data.hdf5"
print(path)
keras_model = load_model(path)

(trainX, trainY), (testX, testY) = mnist.load_data()

trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))

trainX = trainX.astype('float32')
testX = testX.astype('float32')

trainX = trainX / 255.0
testX = testX / 255.0


# Retrieve the tensorflow session
sess = backend.get_session()

# Initialize the Fast Gradient Sign Method (FGSM) attack object and
# use it to create adversarial examples as numpy arrays.
wrap = KerasModelWrapper(keras_model)
fgsm = FastGradientMethod(wrap, sess=sess)
fgsm_params = {'eps': 0.3, 'clip_min': 0., 'clip_max': 1.}

adv_x = fgsm.generate_np(testX, **fgsm_params)

adv_pred = np.argmax(keras_model.predict(adv_x), axis = 1)
adv_acc =  np.mean(np.equal(adv_pred, testY))

for sample in adv_x:
    plt.imshow( sample.reshape((28,28)) , cmap='gray')
    plt.show()

print("The adversarial validation accuracy is: {}".format(adv_acc))