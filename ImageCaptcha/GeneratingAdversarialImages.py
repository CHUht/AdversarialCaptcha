from keras.datasets import cifar10
import random as rd
import numpy as np
import pickle as pkl
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import load_model
from ImageCaptcha.ImageCaptchaPixelAtack import *
import os

# Custom Networks
from OnePixelAttack.networks.lenet import LeNet
from OnePixelAttack.networks.resnet import ResNet

def reshape_to_appropriate(img):
    img = img / 255
    return img.reshape((1, img.shape[1], img.shape[2], 1))


"""
    We need to learn the model 
    This is done to apply the adversarial 
"""
model = "LeNet"
if model == "LeNet":
    pathLeNet = os.getcwd().replace("\ImageCaptcha","")
    pathLeNet = pathLeNet + "/OnePixelAttack/networks/models/lenet.h5"
    lenet = LeNet(pathLeNet)
    model = lenet
elif model == "ResNet":
    pathResNet = os.getcwd().replace("\ImageCaptcha", "")
    pathResNet = pathResNet + "/OnePixelAttack/networks/models/resnet.h5"
    resnet = ResNet(pathResNet)
    model = resnet
else:
    exit()


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

(trainX, trainY), (testX, testY) = cifar10.load_data()

N = len(trainX[0])
Wideness = len(trainX[0]) * 4
Height = len(trainX[0])

"""
    Constructing the Captcha Dataset
    Here we construct a dataset with 60000 Captcha images 
"""
for K in range(60000):

    """
        Constructing the image
        The CAPTCHA will have 4 handwritten digits 
        They will come from the MNIST data set and use adversarial learning to become stronger!
    """

    """
        In this part we choose the four digits that will construct the CAPTCHA
        Choose it by using random choice!
    """
    image_choice = rd.randint(0, len(trainX)-1)

    attack_image = attack(image_choice, model, pixel_count=400, filter=True)

    name = class_names[trainY[image_choice][0]]

    plt.imsave("./400_FILTER_ADVERSARIAL_DATABASE/" + str(K) + "_" + name + ".png", attack_image)

