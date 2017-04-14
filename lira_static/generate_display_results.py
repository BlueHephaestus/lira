"""
This file is mainly for generating human-viewable images from completed predictions, given greyscale images.
Usually run after generate_predictions.py. 
Further documentation can be found in each method.

-Blake Edwards / Dark Element
"""
import os, sys, h5py, pickle, cv2

import numpy as np

import img_handler
from img_handler import get_relative_factor, clear_dir

sys.path.append(os.path.expanduser("~/programming/machine_learning/tuberculosis_project/lira_live/"))

import subsection_handler
from subsection_handler import get_next_overlay_subsection

import post_processing

def generate_display_results(img_archive_dir = "../lira/lira1/data/greyscales.h5", predictions_archive_dir = "../lira/lira1/data/predictions.h5", classification_metadata_dir = "classification_metadata.pkl", results_dir = "results", epochs = 1):
    """
    Arguments:
        img_archive_dir: a string filepath of the .h5 file where the images / greyscales are stored.
        predictions_archive_dir: a string filepath of the .h5 file where the model's predictions on the images/greyscales is stored.
        classification_metadata_dir: a string filepath of the .pkl file where the model's classification strings and color key for each classification are stored.
        results_dir: a string filepath to store each result image.
        epochs: the number of iterations to run our denoising algorithm on each full predictions matrix. Defaults to 1.

    Returns:
        Loops through each image in the `img_archive_dir`, as well as each prediction in the `predictions_archive_dir`. There should be one prediction for each image.
        Denoises the predictions for the image,
        Then loops through each image's subsections, 
        Generates an overlay subsection from the image and it's associated prediction,
        Resizes accordingly,
        And saves the image to the results_dir.

        Has no return values.

    Denoising note: Denoising is no longer done in generate_predictions.py, but in post_processing.py instead!
        We used to do denoising in generate_predictions.py, however this resulted in problems due to how LIRA Live works.
        LIRA-Live, when quitting a session, will get all predictions up to the current slide/subsection, 
            and save these along with any new predictions that were generated. 
        Usually, this just takes a bit of time, but you end up with a file that is the same as it was previously,
            but with the new samples appended to the end.
        However, if the old predictions are modified, then LIRA-Live would save the modified predictions, overwriting any old ones.
        I implemented this before I had a denoiser.
        This means that if I generate new denoised predictions, then open and close LIRA-Live on those new denoised predictions,
            it will save the denoised predictions as our new samples.
        In the case of LIRA-Live we are 100% certain that the data we obtain from it is correct,
            so when we denoise that data before saving it, we inevitably screw up some of our labeled data, polluting our dataset with incorrect labels.
        The solution? Don't use the denoiser when generating predictions.h5. 
        If we instead only use it when generating overlays / display results, or getting statistics (i.e. any other case but LIRA-Live), 
            we avoid this problem. 
        It's also nice, because it means we can experiment with different denoising algorithms far quicker than usual.
        So now, the predictions.h5 is for un-post-processed output, the raw predictions of the network.
        And when end-pipeline statistics or overlays are needed, post_processing.py can be used to prettify them.
        So, in this file, we prettify them with our post_processing.py
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
                Since I haven't yet made a denoising algorithm that works with 3-tensors instead of matrices for predictions,
                    we argmax right here.
                THIS NEEDS TO BE AFTER THE DENOISING, NORMALLY
                """
                img_predictions = np.argmax(img_predictions, axis=2)

                """
                We then denoise our predictions, now that the entire predictions matrix is loaded for this image.
                """
                img_predictions = post_processing.denoise_predictions(img_predictions, len(classifications), epochs)

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
