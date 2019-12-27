from keras.datasets import mnist
import random as rd
import numpy as np
import pickle as pkl
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import load_model
from Captcha.AdaptedOnePixelAttack import *

def reshape_to_appropriate(img):

    img = img / 255
    return img.reshape( (1,img.shape[1],img.shape[2],1) )


"""
    We need to learn the model 
    This is done to apply the adversarial 
"""
model = load_model("./MODELS/LeNet_MNIST_data.hdf5")


digits = [0,1,2,3,4,5,6,7,8,9]

(trainX, trainY), (testX, testY) = mnist.load_data()

N = len(trainX[0])
Wideness = len(trainX[0])*4
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
    list_images = []
    digits_for_file_name = ""
    for i in range(4):

        digit_choice = rd.choice(digits)
        digits_for_file_name = digits_for_file_name + str(digit_choice)
        image_choice = rd.randint(0, len(trainY) - 1)
        while digit_choice != trainY[image_choice]:
            image_choice = rd.randint(0, len(trainY) - 1)

        attack_image = attack(trainX[image_choice],trainY[image_choice],model, pixel_count=300, filter=True)

        list_images.append(attack_image[0])


    """
        In this part we past the values from the images into a new numpy array
    """
    new_image = np.zeros((Height, Wideness))

    for i in range(N):
        for j in range(N):
            new_image[i, j] = list_images[0][i, j]
            new_image[i, j + 28] = list_images[1][i, j]
            new_image[i, j + 28 * 2] = list_images[2][i, j]
            new_image[i, j + 28 * 3] = list_images[3][i, j]


    plt.imsave("./FILTER_ADVERSARIAL_DATABASE/" + str(K) + "_" + digits_for_file_name + ".png", new_image, cmap='Greys')


