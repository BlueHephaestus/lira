"""
This file is used for handling subsections through various helper functions,
    to be used in LIRA's live implementation.

-Blake Edwards / Dark Element
"""

import sys
import numpy as np

sys.path.append("../lira_static/")

import img_handler
from img_handler import *

def get_next_overlay_subsection(img_i, sub_i, factor, img, img_predictions, classifications, colors, alpha=1/3., sub_h=80, sub_w=145):
    """
    Arguments:
        img_i: Index of our img argument in the greyscales archive
        sub_i: Index of our subsection in the img
        factor: Factor to divide our img by
        img: np array of shape (h, w, ...). To reference for obtaining a greyscale subsection for generating our overlay subsection with.
        img_predictions: np array of shape (h//sub_h, w//sub_w). To reference for obtaining a predictions subsection for generating our overlay subsection with.
        classifications: List of strings mapping our class indices to string values, 
            e.g. ["Apple", "Orange", "Banana"...] for 0 -> "Apple", 1 -> "Orange", 2 -> "Banana"
        colors: List of tuples of 3 elements, detailing BGR (B, G, R) colors for each of our 
            class indices / classifications, in the same manner that classifications does.
            -We have to do BGR because that is how OpenCV displays them.
        alpha: Number between 0 and 1, amount of transparency on our overlayed predictions. 
            0 = full transparency of overlay, 
            1 = no transparency of overlay
        sub_h, sub_w: The size of our individual subsections in our image.

    Returns:
        Returns an overlay subsection at the location specified by our factor and sub_i, 
            with the greyscale having colored prediction rectangles of transparency alpha overlaid on top of them.
    """
    """
    Using our division factor, get the row_i and col_i.
    Then get our original greyscale subsection from our img,
    As well as the subsection of predictions for this greyscale subsection, 
        using the same function with slightly modified parameters
    """
    row_i = sub_i//factor
    col_i = sub_i % factor

    greyscale_sub = get_next_subsection(row_i, col_i, img.shape[0], img.shape[1], sub_h, sub_w, img, factor)
    sub_predictions = get_next_subsection(row_i, col_i, img_predictions.shape[0], img_predictions.shape[1], 1, 1, img_predictions, factor)

    """
    Generate an overlay to match our greyscale subsection in height and width, but have 3 values per cell for RGB
    """
    overlay = np.zeros((greyscale_sub.shape[0], greyscale_sub.shape[1], 3))

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
    overlay_sub = add_weighted_overlay(greyscale_sub, overlay, alpha)

    return overlay_sub

def get_prediction_subsection(sub_i, factor, img_predictions):
    """
    Arguments:
        sub_i: Index of our subsection in the img
        factor: Factor to divide our img by
        img_predictions: np array of shape (h//sub_h, w//sub_w). To reference for obtaining a predictions subsection for generating our overlay subsection with.

    Returns:
        Gets subsection of predictions from the location in img_predictions specified by sub_i and factor. Returns this.
    """
    row_i = sub_i//factor
    col_i = sub_i % factor
    sub_predictions = get_next_subsection(row_i, col_i, img_predictions.shape[0], img_predictions.shape[1], 1, 1, img_predictions, factor)
    return sub_predictions

def update_prediction_subsection(sub_i, factor, img_predictions, prediction_sub):
    """
    Arguments:
        sub_i: Index of our subsection in the img
        factor: Factor to divide our img by
        img_predictions: np array of shape (h//sub_h, w//sub_w). To reference for obtaining a predictions subsection for generating our overlay subsection with.
        prediction_sub: Subsection of predictions, usually obtained from get_prediction_subsection. Size may vary depending on factor.

    Returns:
        img_predictions, with the subsection specified by sub_i and factor replaced with prediction_sub.
        Use this for updating only a subsection of our big img_predictions matrix with the new, updated subsection.

        Note: We use the same technique (albeit simplified) from inside get_next_subsection()
    """
    row_i = sub_i//factor
    col_i = sub_i % factor

    img_predictions_h = img_predictions.shape[0]
    img_predictions_w = img_predictions.shape[1]

    prediction_sub_h = prediction_sub.shape[0]
    prediction_sub_w = prediction_sub.shape[1]

    row = row_i * prediction_sub_h
    col = col_i * prediction_sub_w

    img_predictions[row:row+prediction_sub_h, col:col+prediction_sub_w] = prediction_sub
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
