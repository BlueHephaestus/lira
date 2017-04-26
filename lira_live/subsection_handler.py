"""
This file is used for handling subsections through various helper functions,
    to be used in LIRA's live implementation.

-Blake Edwards / Dark Element
"""
"""
An important note on predictions in LIRA-Live:
We have 2 problems:
    1. We want to display only one color in our gui for each prediction, even though each entry is a vector of probabilities
    2. We also need to retain the link to our predictions_hf.get, so that
        predictions in the file can be easily updated once they are corrected using the GUI tool.
So if we just normally argmaxed over img_predictions, we'd break #2 and no longer have a link back to our file.
But if we left it as is, we'd not have an easy way to display the predictions.

The solution I came up with was to argmax over the prediction subsections as they are obtained for display,
    and convert these prediction subsections to one-hots (essentially the inverse of an argmax) 
    in order to update the predictions in the original array.
Those changes can be seen in this file.
"""

import sys
import numpy as np

sys.path.append("../lira_static/")

import img_handler
from img_handler import *

from keras.utils import np_utils

def get_next_overlay_subsection(img_i, sub_i, factor, img, img_predictions, classifications, colors, alpha=1/3., sub_h=80, sub_w=145, rgb=False):
    """
    Arguments:
        img_i: Index of our img argument in the images archive
        sub_i: Index of our subsection in the img
        factor: Factor to divide our img by
        img: np array of shape (h, w, ...). To reference for obtaining a image subsection for generating our overlay subsection with.
        img_predictions: np array of shape (h//sub_h, w//sub_w, class_n). To reference for obtaining a predictions subsection for generating our overlay subsection with.
        classifications: List of strings mapping our class indices to string values, 
            e.g. ["Apple", "Orange", "Banana"...] for 0 -> "Apple", 1 -> "Orange", 2 -> "Banana"
        colors: List of tuples of 3 elements, detailing BGR (B, G, R) colors for each of our 
            class indices / classifications, in the same manner that classifications does.
            -We have to do BGR because that is how OpenCV displays them.
        alpha: Number between 0 and 1, amount of transparency on our overlayed predictions. 
            0 = full transparency of overlay, 
            1 = no transparency of overlay
        sub_h, sub_w: The size of our individual subsections in our image.
        rgb: Boolean for if we are handling rgb images (True), or grayscale images (False).

    Returns:
        Returns an overlay subsection at the location specified by our factor and sub_i, 
            with the image having colored prediction rectangles of transparency alpha overlaid on top of them.
    """
    """
    Using our division factor, get the row_i and col_i.
    Then get our original image subsection from our img,
    As well as the subsection of predictions for this image subsection, 
        using the same function with slightly modified parameters
    """
    row_i = sub_i//factor
    col_i = sub_i % factor

    img_sub = get_next_subsection(row_i, col_i, img.shape[0], img.shape[1], sub_h, sub_w, img, factor)
    sub_predictions = get_next_subsection(row_i, col_i, img_predictions.shape[0], img_predictions.shape[1], 1, 1, img_predictions, factor)

    """
    Argmax over these since we are using it for display
    """
    sub_predictions = np.argmax(sub_predictions, axis=2)

    """
    Generate an overlay to match our image subsection in height and width, but have 3 values per cell for RGB
    """
    overlay = np.zeros((img_sub.shape[0], img_sub.shape[1], 3))

    """
    Generate our rectangles on our overlay
    """
    for prediction_row_i, prediction_row in enumerate(sub_predictions):
        for prediction_col_i, prediction in enumerate(prediction_row):
            prediction = int(prediction)
            color = colors[prediction]
            cv2.rectangle(overlay, (prediction_col_i*sub_w, prediction_row_i*sub_h), (prediction_col_i*sub_w+sub_w, prediction_row_i*sub_h+sub_h), color, -1)

    """
    Generate our weighted overlay subsection, and return it.
    """
    overlay_sub = add_weighted_overlay(img_sub, overlay, alpha, rgb=rgb)

    return overlay_sub

def get_prediction_subsection(sub_i, factor, img_predictions):
    """
    Arguments:
        sub_i: Index of our subsection in the img
        factor: Factor to divide our img by
        img_predictions: np array of shape (h//sub_h, w//sub_w, class_n). 

    Returns:
        Gets subsection of predictions from the location in img_predictions specified by sub_i and factor. 
        Then argmaxes over these predictions, since they are used for display, and
        Returns this.
    """
    row_i = sub_i//factor
    col_i = sub_i % factor
    sub_predictions = get_next_subsection(row_i, col_i, img_predictions.shape[0], img_predictions.shape[1], 1, 1, img_predictions, factor)

    """
    Argmax over these since we are using it for display
    """
    sub_predictions = np.argmax(sub_predictions, axis=2)
    return sub_predictions

def update_prediction_subsection(sub_i, factor, img_predictions, prediction_sub):
    """
    Arguments:
        sub_i: Index of our subsection in the img
        factor: Factor to divide our img by
        img_predictions: np array of shape (h//sub_h, w//sub_w, class_n). 
        prediction_sub: Subsection of predictions, usually obtained from get_prediction_subsection. Size may vary depending on factor.
            only has 2 dimensions, as opposed to img_predictions, which has three.

    Returns:
        img_predictions, with the subsection specified by sub_i and factor replaced with prediction_sub.
        Use this for updating only a subsection of our big img_predictions matrix with the new, updated subsection.

        Since prediction_sub is 2 dimensions, with each entry being the index of a classification/prediction,
            we have to create one-hot / categorical vectors for each of these to store in our full img_predictions matrix.

    """
    row_i = sub_i//factor
    col_i = sub_i % factor

    img_predictions_h = img_predictions.shape[0]
    img_predictions_w = img_predictions.shape[1]

    prediction_sub_h = prediction_sub.shape[0]
    prediction_sub_w = prediction_sub.shape[1]

    row = row_i * prediction_sub_h
    col = col_i * prediction_sub_w

    """
    We convert our prediction_sub into a categorical_prediction_sub. 
        This takes a matrix of shape (n, m),
            where each entry is the index value,
        And converts to a 3-tensor of shape (n, m, class_n), where each entry is the one-hot representation of that index value.
    It does this by first using keras's np_utils.to_categorical method to convert our prediction_sub to a matrix of shape (n*m, class_n),
        using the last dimension of our img_predictions tensor, which will be class_n,
    Then reshapes this into a 3-tensor of shape (n, m, 7), which is of the proper dimensionality to be assigned to our img_predictions
    """
    categorical_prediction_sub = np.reshape(np_utils.to_categorical(prediction_sub, img_predictions.shape[-1]), (prediction_sub_h, prediction_sub_w, -1))
    img_predictions[row:row+prediction_sub_h, col:col+prediction_sub_w] = categorical_prediction_sub
    return img_predictions
    
def list_find(l, e):
    """
    Arguments:
        l: a list
        e: an element to be searched for.

    Returns:
        The index of the element e in the list, if found,
        And -1 if not found.
    """
    for i in range(len(l)):
        if e == l[i]:
            return i
    return -1

def get_sub_max_n(img_hf):
    """
    Arguments:
        img_hf: A dataset to open and iterate through the images in it.

    Returns:
        Loops through all of our images in img_hf to get the max number of subsections for any given image, 
            and returns this number.
    """
    img_n = len(img_hf.keys())
    sub_max_n = 0
    for img_i in range(img_n):
        img = img_hf.get(str(img_i))
        factor = get_relative_factor(img.shape[0], None)
        sub_n = factor**2
        if sub_max_n < sub_n:
            sub_max_n = sub_n
    return sub_max_n

def all_predictions_empty(prediction_sub, classifications):
    """
    Arguments:
        prediction_sub: A subsection of predictions, however could be the entire predictions array. A matrix/np array of integers for classifications/predictions.
        classifications: List of strings mapping our class indices to string values, 
            e.g. ["Apple", "Orange", "Banana"...] for 0 -> "Apple", 1 -> "Orange", 2 -> "Banana"
            Where one of them is "Empty Slide", our empty classification.

    Returns:
        We check if all of our predictions are the "Empty Slide" classification.
            If so, we return True.
            If not, we return False.
    """
    """
    We first find the prediction index to check for, by looking for what classification in our list is the "Empty Slide" classification.
        If we don't have an empty slide classification, which should not happen, then we exit.
    """
    empty_i = list_find(classifications, "Empty Slide")
    if empty_i == -1: sys.exit("Classifications do not include 'Empty Slide' classification.")

    """
    Now that we have a matrix to check, and an integer value (index) to check with, we can simply return an evaluated boolean expression.
    """
    return np.all(prediction_sub==empty_i)

def get_extra_empty_samples(classification_n, empty_classification_n, classifications):
    """
    Arguments:
        classification_n: Total number of classifications we have.
        empty_classification_n: Total number of empty classifications we have.
        classifications: List of strings mapping our class indices to string values, 
            e.g. ["Apple", "Orange", "Banana"...] for 0 -> "Apple", 1 -> "Orange", 2 -> "Banana"
            Where one of them is "Empty Slide", our empty classification.
    Returns:
        Using our formula detailed below, we return the number of extra empty samples to subsequently return.
    """
    """
    We compute the number of extra empty samples using the formula:
                (C-C_e)/n
    Where C is classifications_n, C_e is empty_classifications_n, and n is the number of classes.

    This formula will subtract the number of samples that are already empty classifications from the total,
        and then get a number that is 1/n times that result, so that we always end up with the perfect amount
        of empty samples to add so that we don't disturb/bias our distribution of samples in our training data.
    """
    extra_empty_sample_n = int((classification_n-empty_classification_n)/float(len(classifications)))

    return extra_empty_sample_n
