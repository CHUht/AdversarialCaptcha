# Python Libraries

import pickle
import numpy as np
import pandas as pd
import matplotlib
from keras.datasets import cifar10
from keras import backend as K
import matplotlib.pyplot as plt
import os
from PIL import Image

# Custom Networks
from OnePixelAttack.networks.lenet import LeNet
from OnePixelAttack.networks.resnet import ResNet

# Helper functions
from OnePixelAttack.differential_evolution import differential_evolution
from OnePixelAttack import helper

""" 
    Chose Database 
    Declare lists to use
    And class names for the categorical conversion 
"""
path = "./400_FILTER_ADVERSARIAL_DATABASE/"
x_test = []
y_test = []
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

for img in os.listdir(path):

    # Open and append image in file
    x = Image.open(path + img)
    x = x.convert('RGB')
    plt.imshow(x)
    plt.show()
    x = np.asarray(x)
    x_test.append(x)

    y = []
    # Find the label of the image from the title
    flag = 0
    classification = ""
    for letter in img:
        if letter == ".":
            break
        if letter == "_":
            flag = 1
            continue
        if flag == 0:
            continue
        classification = classification + letter

    print(classification)
    # Set the correct class position to 1!!!!
    y.append(class_names.index(classification))
    y_test.append(y)