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

    # Set the correct class position to 1!!!!
    y.append(class_names.index(classification))
    y_test.append(y)

x_test = np.asarray(x_test)
y_test = np.asarray(y_test)


""" 
    Opening the models 
"""
pathLeNet = os.getcwd().replace("\ImageCaptcha","")
pathLeNet = pathLeNet + "/OnePixelAttack/networks/models/lenet.h5"
lenet = LeNet(pathLeNet)
pathResNet = os.getcwd().replace("\ImageCaptcha", "")
pathResNet = pathResNet + "/OnePixelAttack/networks/models/resnet.h5"
resnet = ResNet(pathResNet)
models = [lenet, resnet]

"""
    Testing the models
"""
network_stats, correct_imgs = helper.evaluate_models(models, x_test, y_test)
correct_imgs = pd.DataFrame(correct_imgs, columns=['name', 'img', 'label', 'confidence', 'pred'])
network_stats = pd.DataFrame(network_stats, columns=['name', 'accuracy', 'param_count'])
print(network_stats)