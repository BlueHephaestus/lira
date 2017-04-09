"""
This file is mainly for generating human-viewable images from completed predictions, given greyscale images.
Usually run after generate_predictions.py. 
Further documentation can be found in each method.

-Blake Edwards / Dark Element
"""
import os, sys, h5py, pickle, cv2

import img_handler
from img_handler import get_relative_factor, clear_dir

sys.path.append(os.path.expanduser("~/programming/machine_learning/tuberculosis_project/lira_live/"))

import subsection_handler
from subsection_handler import get_next_overlay_subsection

def generate_display_results(img_archive_dir = "../lira/lira1/data/greyscales.h5", predictions_archive_dir = "../lira/lira1/data/predictions.h5", classification_metadata_dir = "classification_metadata.pkl", results_dir = "results"):
    """
    Arguments:
        img_archive_dir: a string filepath of the .h5 file where the images / greyscales are stored.
        predictions_archive_dir: a string filepath of the .h5 file where the model's predictions on the images/greyscales is stored.
        classification_metadata_dir: a string filepath of the .pkl file where the model's classification strings and color key for each classification are stored.
        results_dir: a string filepath to store each result image.

    Returns:
        Loops through each image in the `img_archive_dir`, as well as each prediction in the `predictions_archive_dir`. There should be one prediction for each image.
        Then loops through each image's subsections, 
        Generates an overlay subsection from the image and it's associated prediction,
        Resizes accordingly,
        And saves the image to the results_dir.

        Has no return values.
    """

    """
    Clear our results directory of any pre-existing images/data
    """
    clear_dir(results_dir)

    """
    Open our image and prediction files for iteration
    """
    with h5py.File(img_archive_dir, "r") as img_hf:
        with h5py.File(predictions_archive_dir, "r") as predictions_hf:

            """
            Get image number for iteration
            """
            img_n = len(img_hf.keys())

            """
            Get our classification metadata for generating the overlay subsection later.
            """
            f = open(classification_metadata_dir, "r")
            classifications, colors = pickle.load(f)
            f.close()

            """
            Start looping through images
            """
            for img_i in range(img_n):
                """
                Get the image and predictions for the image
                """
                img = img_hf.get(str(img_i))
                img_predictions = predictions_hf.get(str(img_i))

                """
                Get the factor to use for both subsections and resizing
                """
                factor = get_relative_factor(img.shape[0], None)
                sub_n = factor**2

                """
                Start looping through subsections
                """
                for sub_i in range(sub_n):
                    sys.stdout.write("\rIMAGE %i, SUBSECTION %i" % (img_i, sub_i))
                    sys.stdout.flush()

                    """
                    Get row index and column index given our subsection index and factor
                    """
                    row_i = sub_i//factor
                    col_i = sub_i % factor

                    """
                    Get our resize / scale factor from our division factor
                    """
                    resize_factor = 1./factor

                    """
                    Get our overlay subsection
                    """
                    overlay_sub = get_next_overlay_subsection(img_i, sub_i, factor, img, img_predictions, classifications, colors, alpha=0.33, sub_h=80, sub_w=145)

                    """
                    Resize the overlay subsection with our resize_factor
                    """
                    overlay_sub = cv2.resize(overlay_sub, (0,0), fx=resize_factor, fy=resize_factor)

                    """
                    Finally, write the image
                    """
                    cv2.imwrite('%s/%i_%i_%i.jpg' % (results_dir, img_i, row_i, col_i), overlay_sub)
