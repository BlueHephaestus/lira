"""
This file is used for handling subsections through various helper functions,
to be used in LIRA's live implementation.

-Blake Edwards / Dark Element
"""

import sys
import numpy as np

sys.path.append("../slide_testing/")

import img_handler
from img_handler import *

def get_next_overlay_subsection(img_i, sub_i, factor, img, img_predictions, classifications, colors, alpha=1/3., sub_h=80, sub_w=145):
    """
    Given our image index and subsection index, as well as our image and predictions and metadata, 
        we return the corresponding subsection from the corresponding image, with the overlay overlaid with `alpha` transparency.
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
    Given the subsection index, the division factor, and predictions for an image
        we return the predictions from the correct location in the image.
    """
    row_i = sub_i//factor
    col_i = sub_i % factor
    sub_predictions = get_next_subsection(row_i, col_i, img_predictions.shape[0], img_predictions.shape[1], 1, 1, img_predictions, factor)
    return sub_predictions

def update_prediction_subsection(sub_i, factor, img_predictions, prediction_sub):
    """
    Given the subsection index, the division factor, predictions for an image, and a subsection of predictions
        we update the subsection of predictions inside the main predictions for the entire image, 
        using the same technique (albeit simplified) from inside get_next_subsection()
    """
    row_i = sub_i//factor
    col_i = sub_i % factor

    img_predictions_h = img_predictions.shape[0]
    img_predictions_w = img_predictions.shape[1]

    prediction_sub_h = prediction_sub.shape[0]
    prediction_sub_w = prediction_sub.shape[1]
    #sub_img_h = img_predictions.shape[0]//factor
    #sub_img_w = img_predictions.shape[0]//factor

    #row = row_i * sub_img_h
    #col = col_i * sub_img_w
    row = row_i * prediction_sub_h
    col = col_i * prediction_sub_w

    img_predictions[row:row+prediction_sub_h, col:col+prediction_sub_w] = prediction_sub
    return img_predictions
    
def list_find(l, e):
    """
    Given a list l find index of element e, return -1 if not found.
        This is just a small, super helpful function for all_predictions_empty to use.
    """
    for i in range(len(l)):
        if e == l[i]:
            return i
    return -1

def all_predictions_empty(prediction_sub, classifications):
    """
    Given our predictions subsection, and our classifications,
        we check if all of our predictions are the "Empty Slide" classification.
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

def get_extra_empty_samples(classification_n, empty_classification_n, classifications, sample_h, sample_w):
    """
    Given the total number of new classifications `classifications_n`,
        and the total number of new "Empty Slide" classifications `empty_classifications_n`,
    Compute the number of extra empty samples to subsequently return.
    """
    """
    First, compute the number of extra empty samples using the formula:
                (C-C_e)/n
    Where C is classifications_n, C_e is empty_classifications_n, and n is the number of classes.

    This formula will subtract the number of samples that are already empty classifications from the total,
        and then get a number that is 1/n times that result, so that we always end up with the perfect amount
        of empty samples to add so that we don't disturb/bias our distribution of samples in our training data.
    """
    extra_empty_sample_n = int((classification_n-empty_classification_n)/float(len(classifications)))

    """
    Then, generate the empty samples using the same method we used in generate_empty_slide_data, and return.
    """
    """
    sample_mean = 244
    sample_stddev = 0.22

    return np.random.randn(extra_empty_sample_n, sample_h, sample_w)*sample_stddev + sample_mean
    """
    return extra_empty_sample_n
