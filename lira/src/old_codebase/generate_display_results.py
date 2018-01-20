"""
This file is mainly for generating human-viewable images from completed predictions, given greyscale images.
Usually run after generate_predictions.py. 
Further documentation can be found in each method / main function documentation.

-Blake Edwards / Dark Element
"""
import os, sys, h5py, pickle, cv2

import numpy as np

import img_handler
from img_handler import *

import post_processing

def generate_display_results(img_archive_dir = "../lira/lira1/data/greyscales.h5", predictions_archive_dir = "../lira/lira1/data/predictions.h5", classification_metadata_dir = "classification_metadata.pkl", results_dir = "results", alpha=.33, sub_h=80, sub_w=145, neighbor_weight=0.8, rgb=False):
    """
    Arguments:
        img_archive_dir: a string filepath of the .h5 file where the images / greyscales are stored.
        predictions_archive_dir: a string filepath of the .h5 file where the model's predictions on the images/greyscales is stored.
        classification_metadata_dir: a string filepath of the .pkl file where the model's classification strings and color key for each classification are stored.
        results_dir: a string filepath to store each result image.
        alpha: transparency weight of our overlay, percentage b/w 0 and 1, with 0 being no overlay and 1 being only overlay.
        sub_h, sub_w: The size of our individual subsections.
        neighbor_weight: 
            How much importance to put on the neighbor values. 
            This could also be thought of as a smoothing factor.
            Should be between 0 and 1
        rgb: Boolean for if we are handling rgb images (True), or grayscale images (False).

    Returns:
        Loops through each image in the `img_archive_dir`, as well as each prediction in the `predictions_archive_dir`. There should be one prediction for each image.
        Denoises the predictions for the image,
        Resizes our image to be small enough so we can open it,
        Creates a colored rectangle on an overlay for each prediction,
        Then combines the overlay and image, using alpha for the weight to put on the overlay.
        This final overlay image is saved to the results_dir.

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
                sys.stdout.write("\rIMAGE %i" % img_i)
                sys.stdout.flush()

                """
                Get the image and predictions for the image
                """
                img = img_hf.get(str(img_i))
                predictions = np.array(predictions_hf.get(str(img_i)))

                """
                We then denoise our predictions, now that all the predictions are loaded for this image.
                """
                predictions = post_processing.denoise_predictions(predictions, neighbor_weight)

                """
                Get the factor to use for resizing
                """
                factor = get_relative_factor(img.shape[0], None)

                """
                Get our resize / scale factor from our division factor
                """
                resize_factor = 1./factor

                """
                So we want to generate the entire overlay at once. 
                In order to do this, we can't actually make it at full resolution and THEN
                    resize it down to the resized resolution, since
                    often times our images are too big to hold in memory.

                So instead, we want to resize our images down immediately to their final size,
                    and create our overlay at the final size.

                First part's easy, however the second is a bit trickier. 
                
                We want the rectangles in our overlay to be the same size as if we created the
                    entire overlay, and then resized it down by resize_factor.
                Since each rectangle is of size sub_h x sub_w, we can get the resulting size
                    of each subsection via sub_h * resize_factor x sub_w * resize_factor.
                However since we need to draw rectangles with integer values, we then 
                    floor these calculations and then convert to int.
                """
                resized_sub_h = int(np.floor(sub_h * resize_factor))
                resized_sub_w = int(np.floor(sub_w * resize_factor))

                """
                Once we've got that, we're halfway there.
                The reason we do this is because of the following problem. 
               
                Each time we draw a resized subsection rectangle, it will be of size
                    resized_sub_h x resized_sub_w. 
                
                Since we draw the same amount of rectangles as predictions,
                    we will have predictions.shape[0] x predictions.shape[1] rectangles,
                    each one of shape resized_sub_h x resized_sub_w.

                So we know that:
                    the height of our overlay is resized_sub_h * predictions.shape[0]
                    the width of our overlay is  resized_sub_w * predictions.shape[1]

                However since we have to cast our resized_sub_h to integer in the calculation,
                    it means we may slowly end up with an overlay with dimensions which
                    don't match our image. This causes a mismatch between the images 
                    once the entire overlay is created, and it's very ugly. 

                Fortunately we can fix this. Using the height as the example dimension:

                    img.shape[0] * fy = resized_sub_h * predictions.shape[0]

                We can see this is the condition we want to be true. 
                We want the resized dimension(s) of our original image to match
                    the dimension(s) of our prediction overlay.
                
                So from this, we can see we already know everything but fy, 
                    and since it's much easier to solve for fy than to figure out some
                    special way to draw our subsection rectangles, we calculate:

                    fy = (resized_sub_h * predictions.shape[0])/img.shape[0]

                And we repeat that for fx, as well.
                """
                fy = (resized_sub_h * predictions.shape[0]) / float(img.shape[0])
                fx = (resized_sub_w * predictions.shape[1]) / float(img.shape[1])

                """
                We then cast our img to nparray so we can properly resize,
                    and then resize using our calculated fx and fy.
                """
                img = cv2.resize(np.array(img), (0,0), fx=fx, fy=fy)

                """
                In order to have integers for each prediction, 
                    we argmax over the probability vectors for each prediction,
                    giving us an integer. 
                We then cast it as such.
                """
                predictions = np.argmax(predictions, axis=2)
                predictions = predictions.astype(np.uint8)

                """
                Initialize our new color overlay to match our img
                """
                overlay = np.zeros((img.shape[0], img.shape[1], 3))

                """
                Then loop through each prediction in our predictions,
                    using the row and col indices with the resized_sub_h and resized_sub_w for the location of our colored rectangle on our overlay,
                    and use the prediction's value in our color key for the color of this rectangle.
                """
                for prediction_row_i, prediction_row in enumerate(predictions):
                    for prediction_col_i, prediction in enumerate(prediction_row):
                        color = colors[prediction]
                        cv2.rectangle(overlay, (prediction_col_i*resized_sub_w, prediction_row_i*resized_sub_h), (prediction_col_i*resized_sub_w+resized_sub_w, prediction_row_i*resized_sub_h+resized_sub_h), color, -1)
                
                """
                We then add the img and overlay together into a weighted overlay using our alpha and function for it.
                """
                overlay = add_weighted_overlay(img, overlay, alpha, rgb=rgb)
                cv2.imwrite("%s%s%i.jpg" % (results_dir, os.sep, img_i), overlay)
