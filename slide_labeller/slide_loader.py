"""
This file is in charge of loading a new subsection for labelling each time the get_next_sub() function is called.
    it makes sure to give it according to the size of the subsection specified, and give both the greyscale full resolution and the combined image.

This way, we have an easy interface for our GUI tools to use.

-Blake Edwards / Dark Element
"""

import sys, os, json, h5py
import numpy as np

import slide_handler_base
from slide_handler_base import *

sys.path.append("../slide_testing/")

import img_handler
from img_handler import *

sub_h = 80
sub_w = 145

def get_next_subsection(img_i, sub_i, img_archive_dir):
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

        #Get our info via our factor
        factor = get_relative_factor(img.shape[0], None)
        sub_n = factor**2
        row_i = sub_i//factor
        col_i = sub_i % factor

        #Get the next subsection
        next_sub = img_handler.get_next_subsection(row_i, col_i, img.shape[0], img.shape[1], sub_h, sub_w, img, factor)
        return next_sub

img_archive_dir = "../lira/lira1/data/greyscales.h5"
predictions_archive_dir = "../lira/lira1/data/predictions.h5"

"""
We don't care what image this is. we just care about our next subsection.
    However, we have our subsections stored according to their image.


for a in range(8):
    img = get_next_subsection(a, 0, img_archive_dir)
    disp_img_fullscreen(img)
"""
