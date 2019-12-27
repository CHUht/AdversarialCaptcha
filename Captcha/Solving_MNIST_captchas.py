import os
from PIL import Image
from PIL import ImageOps
import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.models import load_model

""" 
    Start by loading the model
    And the type of filter to use!
    We can load several different models here
"""
model_choice = "LeNet Adversarial"
noise_filter = "Median"
if model_choice == "DP":
    model = load_model("./MODELS/captcha_model.hdf5")
elif model_choice == "LeNet":
    model = load_model("./MODELS/LeNet_MNIST_data.hdf5")
elif model_choice == "DP Adversarial" :
    model = load_model("./MODELS/adversarial_captcha_model.hdf5")
elif model_choice == "LeNet Median":
    model = load_model("./MODELS/LeNet_Median_Filter.hdf5")
elif model_choice == "ResNet":
    model = load_model("./MODELS/ResNet_MNIST.hdf5")
elif model_choice == "LeNet Adversarial":
    model = load_model("./MODELS/LeNet_Adversarial_Training.hdf5")
else:
    exit()

path = "./FILTER_ADVERSARIAL_DATABASE/"

"""
    Here we loop through all the images in the database
    We them try to see how many our model gets right
    By using an accuracy metric
"""
acc = 0
listing = os.listdir(path)
k = 0
for image_name in listing:

    """
        Start by reading the image
        And converting to grayscale
    """
    img = cv2.imread(path + image_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    """
        Here we have the addition og noise removing filters
        Let's see how the model performs under the condition of noise removal
    """
    if noise_filter == "Blur":
        img = cv2.blur(img,(3,3))
    elif noise_filter == "Gaussian":
        img = cv2.GaussianBlur(img, (3,3), 0)
    elif noise_filter == "Median":
        img = cv2.medianBlur(img, 3)
    else:
        pass

    """²
        Separating the digits 
        Here they are separated by their position!
        Afterwards we append the predicted solution with the Captcha
    """
    predicted_captcha = []
    x = 0
    while x != 112:
        letter_image = img[0:28, x:x + 28]
        x = x + 28

        # Prepare the data for the deep learning model
        letter_image = 255 - letter_image
        letter_image = letter_image / 255
        letter_image = letter_image.reshape((1, 28, 28, 1))


        predictions = model.predict(letter_image)
        predicted_captcha.append(np.argmax(predictions))


    """
        Here we extract the correct captcha from the image name
        This is done to compare right answers and analyse the accuracy
    """
    real_captcha = []
    flag = 0
    for letter in image_name:
        if letter == ".":
            break
        if letter == "_":
            flag = 1
            continue
        if flag == 0:
            continue
        flag = 1
        real_captcha.append(int(letter))

    if predicted_captcha == real_captcha :
        acc = acc + 1
    k = k + 1
    print(acc/k)


acc = acc/len(listing)
print(acc)

