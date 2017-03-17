import sys, h5py, pickle, cv2
import numpy as np


sys.path.append("../../markov_random_fields/MRF/")

import mrf_denoiser

sys.path.append("../slide_testing/")

import img_handler
from img_handler import *

sys.path.append("../slide_labeller/")

import subsection_handler
from subsection_handler import *

class_n = 7

img_archive_dir="../lira/lira1/data/greyscales.h5"
predictions_archive_dir="../lira/lira1/data/predictions.h5"
classification_metadata_dir="../slide_testing/classification_metadata.pkl"

results_dir = "../slide_testing/results"

with h5py.File(img_archive_dir, "r") as img_hf:
    with h5py.File(predictions_archive_dir, "r+") as predictions_hf:
        img_n = len(img_hf.keys())

        f = open(classification_metadata_dir, "r")
        classifications, colors = pickle.load(f)
        f.close()

        for img_i in range(img_n):
            print img_i
            img = img_hf.get(str(img_i))
            img_predictions = predictions_hf.get(str(img_i))

            resize_factor = 1/float(get_relative_factor(img.shape[0], None))
            #print np.array(img_predictions)
            #print np.array(img_predictions).shape
            #img_predictions = mrf_denoiser.denoise(np.array(img_predictions), class_n)
            """
            ATTENTION
            Apparently just doing img_predictions = "whatever" is not enough, you have to reference the inside elements so as not to break the reference.
            So we do img_predictions[...] = "whatever instead.
            """
            img_predictions[...] = mrf_denoiser.denoise(np.array(img_predictions), class_n)
            print ""
            #predictions_hf.create_dataset(str(img_i), data=img_predictions)#if the first bit doesn't work

            """
            Then we loop through the subs and get each overlay subsection, then save it. 
            This is only temporary until we upgrade generate_overlay to do the same or make our own script that is separate from it.
            """

            factor = get_relative_factor(img.shape[0], None)
            sub_n = factor**2
            for sub_i in range(sub_n):
                row_i = sub_i//factor
                col_i = sub_i % factor

                overlay_sub = get_next_overlay_subsection(img_i, sub_i, factor, img, img_predictions, classifications, colors, alpha=0.33, sub_h=80, sub_w=145)
                overlay_sub = cv2.resize(overlay_sub, (0,0), fx=resize_factor, fy=resize_factor)
                cv2.imwrite('%s/%i_%i_%i.jpg' % (results_dir, img_i, row_i, col_i), overlay_sub)
