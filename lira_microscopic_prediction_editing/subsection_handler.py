"""
This file is used for handling subsections through various helper functions,
    to be used in the microscopic prediction editor's implementation.

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

import cv2
import numpy as np

def fill_overlay(overlay, predictions, sub_h, sub_w, colors):
    """
    Arguments:
        overlay: A zeroed numpy array, of the same shape as our image. 
            This will have our rectangle drawn on it so as to retain the reference to the original variable, and so we don't need to return anything.
        predictions: np array of shape (img_h//sub_h, img_w//sub_w, class_n). A matrix/np array of integers for classifications/predictions.
        sub_h, sub_w: The size of our individual subsections in our image.
        colors: List of tuples of 3 elements, detailing BGR (B, G, R) colors for each of our 
            class indices / classifications, in the same manner that classifications does.
            -We have to do BGR because that is how OpenCV displays them.
    Returns:
        Has no return value, however uses the fact that our overlay is a matrix to fill it with values, 
            so that the passed in argument changes in the caller function.
        Our overlay is filled with rectangles of size sub_h x sub_w, 
            where each rectangle is generated according to a matching integer in our predictions matrix,
            and it's position is determined by the row and column index of each matching integer, such that
                new_row_i = row_i * sub_h , 
                new_col_i = col_i * sub_w
        The prediction is the index to the respective color in the colors list.
    """
    for prediction_row_i, prediction_row in enumerate(predictions):
        for prediction_col_i, prediction in enumerate(prediction_row):
            color = colors[prediction]
            cv2.rectangle(overlay, (prediction_col_i*sub_w, prediction_row_i*sub_h), (prediction_col_i*sub_w+sub_w, prediction_row_i*sub_h+sub_h), color, -1)


def to_categorical(vec, width):
    """
    Arguments:
        vec: Vector of indices
        width: Width of our categorical matrix, should be the max value in our vector (but could also be larger if you want).

    Returns:
        A one hot / categorical matrix, so that each entry is the one-hot vector of the index, e.g.
            2 (with width = 4) -> [0, 0, 1, 0]
    """
    categorical_mat = np.zeros((len(vec), width))
    categorical_mat[np.arange(len(vec)), vec] = 1
    return categorical_mat

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

