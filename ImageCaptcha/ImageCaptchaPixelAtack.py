# Python Libraries

import pickle
import numpy as np
import pandas as pd
import matplotlib
from keras.datasets import cifar10
from keras import backend as K
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os

# Custom Networks
from OnePixelAttack.networks.lenet import LeNet
from OnePixelAttack.networks.pure_cnn import PureCnn
from OnePixelAttack.networks.network_in_network import NetworkInNetwork
from OnePixelAttack.networks.resnet import ResNet
from OnePixelAttack.networks.densenet import DenseNet
from OnePixelAttack.networks.wide_resnet import WideResNet
from OnePixelAttack.networks.capsnet import CapsNet


# Helper functions
from OnePixelAttack.differential_evolution import differential_evolution
from OnePixelAttack import helper

matplotlib.style.use('ggplot')
np.random.seed(100)

# Load the dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Class names from this dataset!
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Plot an image just to see it!
#helper.plot_image(x_test[image_id])


def perturb_image(xs, img):
    # If this function is passed just one perturbation vector,
    # pack it in a list to keep the computation the same
    if xs.ndim < 2:
        xs = np.array([xs])

    # Copy the image n == len(xs) times so that we can
    # create n new perturbed images
    tile = [len(xs)] + [1] * (xs.ndim + 1)
    imgs = np.tile(img, tile)

    # Make sure to floor the members of xs as int types
    xs = xs.astype(int)

    for x, img in zip(xs, imgs):
        # Split x into an array of 5-tuples (perturbation pixels)
        # i.e., [[x,y,r,g,b], ...]
        pixels = np.split(x, len(x) // 5)
        for pixel in pixels:
            # At each pixel's x,y position, assign its rgb value
            x_pos, y_pos, *rgb = pixel
            img[x_pos, y_pos] = rgb

    return imgs



def predict_classes(xs, img, target_class, model, filter, minimize=True):
    # Perturb the image with the given pixel(s) x and get the prediction of the model
    imgs_perturbed = perturb_image(xs, img)

    # Adding the filter to the prediction in order to make the attack more robust to filtering
    if filter == True:
        imgs_filtered = []
        for img in imgs_perturbed:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img = cv2.medianBlur(img, 3)
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
            imgs_filtered.append(img)
        imgs_perturbed = np.asarray(imgs_filtered)

    predictions = model.predict(imgs_perturbed)[:,target_class]
    # This function should always be minimized, so return its complement if needed
    return predictions if minimize else 1 - predictions


def attack_success(x, img, target_class, model, targeted_attack=False, verbose=False):
    # Perturb the image with the given pixel(s) and get the prediction of the model
    attack_image = perturb_image(x, img)

    confidence = model.predict(attack_image)[0]
    predicted_class = np.argmax(confidence)

    # If the prediction is what we want (misclassification or
    # targeted classification), return True
    if verbose:
        print('Confidence:', confidence[target_class])
    if ((targeted_attack and predicted_class == target_class) or
            (not targeted_attack and predicted_class != target_class)):
        return True


def attack(img_id, model, target=None, pixel_count=1,
           maxiter=75, popsize=400, verbose=False, filter=False):
    # Change the target class based on whether this is a targeted attack or not
    targeted_attack = target is not None
    target_class = target if targeted_attack else y_train[img_id, 0]

    # Define bounds for a flat vector of x,y,r,g,b values
    # For more pixels, repeat this layout
    bounds = [(0, 32), (0, 32), (0, 256), (0, 256), (0, 256)] * pixel_count

    # Population multiplier, in terms of the size of the perturbation vector x
    popmul = max(1, popsize // len(bounds))

    # Format the predict/callback functions for the differential evolution algorithm
    def predict_fn(xs):
        return predict_classes(xs, x_train[img_id], target_class,
                               model, filter,target is None)

    def callback_fn(x, convergence):
        return attack_success(x, x_train[img_id], target_class,
                              model, targeted_attack, verbose)

    # Call Scipy's Implementation of Differential Evolution
    attack_result = differential_evolution(
        predict_fn, bounds, maxiter=maxiter, popsize=popmul,
        recombination=1, atol=-1, callback=callback_fn, polish=False)

    # Calculate some useful statistics to return from this function
    attack_image = perturb_image(attack_result.x, x_train[img_id])[0]
    prior_probs = model.predict_one(x_train[img_id])
    predicted_probs = model.predict_one(attack_image)
    predicted_class = np.argmax(predicted_probs)
    actual_class = y_train[img_id, 0]
    success = predicted_class != actual_class
    cdiff = prior_probs[actual_class] - predicted_probs[actual_class]

    # Show the best attempt at a solution (successful or not)
    #helper.plot_image(attack_image, actual_class, class_names, predicted_class)

    #return [model.name, pixel_count, img_id, actual_class, predicted_class, success, cdiff, prior_probs,
            #predicted_probs, attack_result.x]
    return attack_image


if __name__ == "__main__":

    """
        Evaluate the networks part!
    """


    pathLeNet = os.getcwd().replace("\ImageCaptcha","")
    pathLeNet = pathLeNet + "/OnePixelAttack/networks/models/lenet.h5"
    LeNet = LeNet(pathLeNet)
    pathResNet = os.getcwd().replace("\ImageCaptcha", "")
    pathResNet = pathResNet + "/OnePixelAttack/networks/models/resnet.h5"
    ResNet = ResNet(pathResNet)

    pathPureCnn = os.getcwd().replace("\ImageCaptcha", "")
    pathPureCnn = pathPureCnn + "/OnePixelAttack/networks/models/pureCnn.h5"
    PureCnn = PureCnn(pathPureCnn)
    PureCnn.train()

    exit()


    models = [LeNet, ResNet, PureCnn]


    network_stats, correct_imgs = helper.evaluate_models(models, x_test, y_test)
    correct_imgs = pd.DataFrame(correct_imgs, columns=['name', 'img', 'label', 'confidence', 'pred'])
    network_stats = pd.DataFrame(network_stats, columns=['name', 'accuracy', 'param_count'])
    print(network_stats)


    """
        Evaluate the images part!
    """

    """
    image_id = 102
    pixels = 30 # Number of pixels to attack
    model = resnet
    attack_image = attack(image_id, model, pixel_count=pixels, filter=True)
    plt.imshow(attack_image)
    plt.show()

    attack_image = cv2.cvtColor(attack_image, cv2.COLOR_BGR2HSV)

    new_image = cv2.medianBlur(attack_image, 3)
    new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)

    plt.imshow(new_image)
    plt.show()
    """

    """
    K = 0
    for image,label in zip(x_train,y_train):

        plt.imshow(image)
        plt.show()

        a = input()

        if( a  == "y"):
            print(class_names[label[0]])
            plt.imsave("./CIFAR_FOR_CAPTCHA/" + str(K) + "_" + class_names[label[0]] + ".png", image)
            K = K + 1
        else:
            pass
    """