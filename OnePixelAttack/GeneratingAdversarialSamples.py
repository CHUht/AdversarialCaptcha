# Python Libraries

import pickle
import numpy as np
import pandas as pd
import matplotlib
from keras.datasets import cifar10
from keras import backend as K

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

lenet = LeNet()
resnet = ResNet()

models = [lenet, resnet]


network_stats, correct_imgs = helper.evaluate_models(models, x_test, y_test)
correct_imgs = pd.DataFrame(correct_imgs, columns=['name', 'img', 'label', 'confidence', 'pred'])
network_stats = pd.DataFrame(network_stats, columns=['name', 'accuracy', 'param_count'])

print(network_stats)

def predict_classes(xs, img, target_class, model, minimize=True):
    # Perturb the image with the given pixel(s) x and get the prediction of the model
    imgs_perturbed = perturb_image(xs, img)
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
           maxiter=75, popsize=400, verbose=False):
    # Change the target class based on whether this is a targeted attack or not
    targeted_attack = target is not None
    target_class = target if targeted_attack else y_test[img_id, 0]

    # Define bounds for a flat vector of x,y,r,g,b values
    # For more pixels, repeat this layout
    bounds = [(0, 32), (0, 32), (0, 256), (0, 256), (0, 256)] * pixel_count

    # Population multiplier, in terms of the size of the perturbation vector x
    popmul = max(1, popsize // len(bounds))

    # Format the predict/callback functions for the differential evolution algorithm
    def predict_fn(xs):
        return predict_classes(xs, x_test[img_id], target_class,
                               model, target is None)

    def callback_fn(x, convergence):
        return attack_success(x, x_test[img_id], target_class,
                              model, targeted_attack, verbose)

    # Call Scipy's Implementation of Differential Evolution
    attack_result = differential_evolution(
        predict_fn, bounds, maxiter=maxiter, popsize=popmul,
        recombination=1, atol=-1, callback=callback_fn, polish=False)

    # Calculate some useful statistics to return from this function
    attack_image = perturb_image(attack_result.x, x_test[img_id])[0]
    prior_probs = model.predict_one(x_test[img_id])
    predicted_probs = model.predict_one(attack_image)
    predicted_class = np.argmax(predicted_probs)
    actual_class = y_test[img_id, 0]
    success = predicted_class != actual_class
    cdiff = prior_probs[actual_class] - predicted_probs[actual_class]

    # Show the best attempt at a solution (successful or not)
    #helper.plot_image(attack_image, actual_class, class_names, predicted_class)

    #return [model.name, pixel_count, img_id, actual_class, predicted_class, success, cdiff, prior_probs,
            #predicted_probs, attack_result.x]
    return attack_image


image_id = 102
pixels = 5 # Number of pixels to attack
model = resnet
attack_image = attack(image_id, model, pixel_count=pixels)
helper.plot_image(attack_image)

new_x_train = []
new_y_train = []
for i in range(5):
    print(i)
    new_x_train.append(x_train[i])
    new_y_train.append(0)
    new_x_train.append(attack(i, model, pixel_count=pixels))
    new_y_train.append(1)

new_x_test = []
new_y_test = []
for i in range(5,10):
    print(i)
    new_x_test.append(x_train[i])
    new_y_test.append(0)
    new_x_test.append(attack(i, model, pixel_count=pixels))
    new_y_test.append(1)

with open("train.pkl","wb") as file:
    pickle.dump(new_x_train,file)
    pickle.dump(new_y_train,file)

with open("test.pkl", "wb") as file:
    pickle.dump(new_x_test,file)
    pickle.dump(new_y_test,file)