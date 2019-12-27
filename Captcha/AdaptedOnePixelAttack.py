from OnePixelAttack.differential_evolution import differential_evolution
from OnePixelAttack import helper
import numpy as np
import matplotlib.pyplot as plt
import cv2

def reshape_to_appropriate(img):

    img = img/255
    return img.reshape( (len(img),img.shape[1],img.shape[2],1) )

def perturb_image(xs, img):
    # If this function is passed just one perturbation vector,
    # pack it in a list to keep the computation the same
    if xs.ndim < 2:
        xs = np.array([xs])

    # Copy the image n == len(xs) times so that we can
    # create n new perturbed images
    imgs = []
    for i in range(len(xs)):
        imgs.append(np.copy(img))
    imgs = np.asarray(imgs)


    # Make sure to floor the members of xs as int types
    xs = xs.astype(int)

    for x,img in zip(xs,imgs):
        # Split x into an array of 3-tuples (perturbation pixels)
        # i.e., [[x,y,r,g,b], ...]
        pixels = np.split(x, len(x) // 3)
        for pixel in pixels:
            # At each pixel's x,y position, assign its rgb value
            x_pos, y_pos, rgb = pixel
            img[x_pos, y_pos] = rgb

    return imgs

def predict_classes(xs, img, target_class, model,filter, minimize=True):
    # Perturb the image with the given pixel(s) x and get the prediction of the model
    imgs_perturbed = perturb_image(xs, img)

    # Adding the filter to the prediction in order to make the attack more robust to filtering
    if filter == True:
        imgs_filtered = []
        for img in imgs_perturbed:
            imgs_filtered.append(cv2.medianBlur(img, 3))
        imgs_perturbed = np.asarray(imgs_filtered)

    reshaped_perturbed_image = reshape_to_appropriate(imgs_perturbed)
    predictions = model.predict(reshaped_perturbed_image)[:,target_class]
    return predictions if minimize else 1 - predictions


def attack_success(x, img, target_class, model, targeted_attack=False, verbose=False):
    # Perturb the image with the given pixel(s) and get the prediction of the model
    attack_image = perturb_image(x, img)
    reshaped_attack_image = reshape_to_appropriate(attack_image)

    confidence = model.predict(reshaped_attack_image)[0]
    predicted_class = np.argmax(confidence)

    # If the prediction is what we want (misclassification or
    # targeted classification), return True
    if verbose:
        print('Confidence:', confidence[target_class])
    if ((targeted_attack and predicted_class == target_class) or
            (not targeted_attack and predicted_class != target_class)):
        return True


def attack(img, label, model, target=None, pixel_count=1,
           maxiter=75, popsize=400, verbose=False, filter=False):

    copy_img = np.copy(img)
    # Change the target class based on whether this is a targeted attack or not
    targeted_attack = target is not None
    target_class = target if targeted_attack else label


    # Define bounds for a flat vector of x,y,r,g,b values
    # For more pixels, repeat this layout
    bounds = [(0, 28), (0, 28), (0, 256)] * pixel_count

    # Population multiplier, in terms of the size of the perturbation vector x
    popmul = max(1, popsize // len(bounds))

    # Format the predict/callback functions for the differential evolution algorithm
    def predict_fn(xs):
        return predict_classes(xs, img, target_class,
                               model, filter)

    def callback_fn(x, convergence):
        return attack_success(x, img, target_class,
                              model, targeted_attack, verbose)

    # Call Scipy's Implementation of Differential Evolution
    attack_result = differential_evolution(
        predict_fn, bounds, maxiter=maxiter, popsize=popmul,
        recombination=1, atol=-1, callback=callback_fn, polish=False)


    # Calculate some useful statistics to return from this function
    attack_image = perturb_image(attack_result.x, copy_img)

    # Show the best attempt at a solution (successful or not)
    #helper.plot_image(attack_image, actual_class, class_names, predicted_class)

    #return [model.name, pixel_count, img_id, actual_class, predicted_class, success, cdiff, prior_probs,
            #predicted_probs, attack_result.x]
    return attack_image
