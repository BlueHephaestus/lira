"""
This file is in charge of loading a new subsection for labelling each time the get_next_img_subsection() function is called.
    it makes sure to give it according to the size of the subsection specified, and give both the greyscale full resolution and the combined image.

This way, we have an easy interface for our GUI tools to use.

-Blake Edwards / Dark Element
"""

import sys, os, json, h5py, pickle
import numpy as np

import slide_handler_base
from slide_handler_base import *

sys.path.append("../slide_testing/")

import img_handler
from img_handler import *

sub_h = 80
sub_w = 145

img_archive_dir = "../lira/lira1/data/greyscales.h5"
predictions_archive_dir = "../lira/lira1/data/predictions.h5"
classification_metadata_dir = "../slide_testing/classification_metadata.pkl"

alpha = 1/3.

#Load our classification metadata
f = open(classification_metadata_dir, "r")
classifications, colors = pickle.load(f)

def get_next_img_subsection(img_i, sub_i, factor, img_archive_dir):
    """
    Given our image index and subsection index, we:
        1. Open up the correct image (they are sorted by index in our .h5 file), img_i
        2. Get the number of subsections and other info for this image via our get_relative_factor function
        3. Plug all our info into our already existing get_next_subsection function we used to generate the overlay
        4. Return this subsection
    We do this every time because we don't want to hold the entire images in memory at once while trying to run our GUI program.
    Just about the same reasons we did it when generating the overlay
    """
    with h5py.File(img_archive_dir, "r") as hf:
        #Get our img
        img = hf.get(str(img_i))

        #Get our info via our division factor
        sub_n = factor**2
        row_i = sub_i//factor
        col_i = sub_i % factor

        #Get the next subsection
        next_sub = get_next_subsection(row_i, col_i, img.shape[0], img.shape[1], sub_h, sub_w, img, factor)
        return next_sub

with h5py.File(img_archive_dir, "r") as img_hf:
    with h5py.File(predictions_archive_dir, "r") as predictions_hf:
        """
        We loop through each image index, with which we can get the entire image and the assorted predictions.
        As we loop through our images, we don't know the number of subsections (sub_n) each has. 
            However, we can use get_relative_factor to get our division factor, since it was the same method we determined it
                when we generated the predictions and overlay. 
            Once we do that, we can loop through the number of subsections we should have, so we only go to the next image in the archive once we are done with the current one.
            We can also only have to calculate the factor once per image, instead of putting the factor calculation in the get_next_subsection function, 
                where it would be calculated once for every subsection. O(1) vs O(n), if n = sub_n
        """
        #Get image number
        img_n = len(img_hf.keys())

        #Start our big loop through images and then subsections
        for img_i in range(img_n):
            img = img_hf.get(str(img_i))
            img_predictions = predictions_hf.get(str(img_i))

            factor = get_relative_factor(img.shape[0], None)
            sub_n = factor**2
            for sub_i in range(sub_n):
                #Get our original greyscale subsection
                greyscale_sub = get_next_img_subsection(img_i, sub_i, factor, img_archive_dir)

                #Generate an overlay to match our greyscale subsection in height and width, but have 3 values per cell for RGB
                overlay = np.zeros((greyscale_sub.shape[0], greyscale_sub.shape[1], 3))

                """
                Now that we have our original greyscale subection, 
                    we need the subsection with our overlay on top of it.
                For this, we have the predictions archive. 
                We also have the coolest function ever, get_next_subsection (and the guy who made it is pretty cool too),
                    which we can use for this as well - as long as we increase our row_i and col_i appropriately.
                """
                row_i = sub_i//factor
                col_i = sub_i % factor
                sub_predictions = get_next_subsection(row_i, col_i, img_predictions.shape[0], img_predictions.shape[1], 1, 1, img_predictions, factor)

                """
                OI FUTURE SELF
                we have it saving and loading and displaying correctly, time for next step!
                
                We are waiting to see if Bryce is ok with the one-image idea
                Good luck, have fun ^-^ o7
                """
                #Generate our rectangles on our overlay
                for prediction_row_i, prediction_row in enumerate(sub_predictions):
                    for prediction_col_i, prediction in enumerate(prediction_row):
                        prediction = int(prediction)
                        color = colors[prediction]
                        cv2.rectangle(overlay, (prediction_col_i*sub_w, prediction_row_i*sub_h), (prediction_col_i*sub_w+sub_w, prediction_row_i*sub_h+sub_h), color, -1)

                overlay_sub = add_weighted_overlay(greyscale_sub, overlay, alpha)

                #disp_img_fullscreen(overlay)
                #disp_img_fullscreen(overlay_sub)
