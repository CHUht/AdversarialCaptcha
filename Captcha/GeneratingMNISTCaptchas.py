from keras.datasets import mnist
import random as rd
import numpy as np
import pickle as pkl
from PIL import Image
import matplotlib.pyplot as plt

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
        image_choice = rd.randint(0,len(trainY)-1)
        while digit_choice != trainY[image_choice]:
            image_choice = rd.randint(0, len(trainY)-1)

        list_images.append(np.asarray(trainX[image_choice]))

    """
        In this part we past the values from the images into a new numpy array
    """
    new_image = np.zeros((Height, Wideness))

    for i in range(N):
        for j in range(N):

            new_image[i,j] = list_images[0][i,j]
            new_image[i, j + 28] = list_images[1][i,j]
            new_image[i, j + 28*2] = list_images[2][i, j]
            new_image[i, j + 28*3] = list_images[3][i, j]

    #pkl.dump(new_image, open("./DATABASE/" + str(K) + "_" + digits_for_file_name, "wb"))
    plt.imsave("./DATABASE/" + str(K) + "_" + digits_for_file_name + ".png", new_image, cmap='Greys')







