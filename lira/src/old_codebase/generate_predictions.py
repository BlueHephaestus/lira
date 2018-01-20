import sys, os, time
import numpy as np
import h5py, pickle
import cv2

import static_config
from static_config import StaticConfig

import img_handler
from img_handler import *

"""
TEMP
"""
import post_processing

def generate_predictions(model_1, model_2, model_dir="../lira/lira2/saved_networks", img_archive_dir = "../lira/lira1/data/greyscales.h5", predictions_archive_dir = "../lira/lira1/data/predictions.h5", rects_archive_dir = "../lira/lira1/data/bounding_rects.pkl", classification_metadata_dir = "classification_metadata.pkl", rgb=False):
    """
    Arguments:
        model_1: String containing the filename of where our first model is stored, to be used for classifying type 1 images and obtaining predictions
        model_2: String containing the filename of where our second model is stored, to be used for classifying type 2 and 3 images and obtaining predictions
        model_dir: Directory of where all our models are stored. Will be used with `model_1`, `model_2`, and `object_detection_model` to obtain where the model is now located 
        img_archive_dir: a string filepath of the .h5 file where the images / greyscales are stored.
        rects_archive_dir: a string filepath of the .pkl file where the detected rects are stored.
        predictions_archive_dir: a string filepath of the .h5 file where the model's predictions on the images/greyscales will be stored.
        classification_metadata_dir: a string filepath of the .pkl file where the model's classification strings and color key for each classification will be stored.
        rgb: Boolean for if we are handling rgb images (True), or grayscale images (False).

    Returns:
        Loads bounding rectangles for detected Type 1 classifications,
        Goes through each image, 
        Then goes through all the individual subsections of size sub_hxsub_w in the image.
        If a subsection is inside a bounding rectangle, it is classified with the first classification model,
            otherwise it is classified with the second classification model.

        Each of these subsections gives us a prediction vector as output (regardless of the model used),
            and this is stored into a global array for all predictions.
        These predictions are written to `predictions_archive_dir`
        
        Has no return value.

    Denoising note: Denoising is no longer done in this file, but in post_processing.py instead!
        We used to do denoising in this file, however this resulted in problems due to how LIRA Live works.
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
    """

    """
    This is untested with other sizes of subsections.
    However, I have trouble finding a reason why it wouldn't work with other sizes.
    That is, unless those sizes were really weird (e.g. 0, -1, .74, etc)
    """
    sub_h = 80
    sub_w = 145

    """
    Mini batch size
    """
    mb_n = 70

    """
    Enable this if you want to see how long it took to classify each image
        *cough* if you want to show off *cough*
    """
    print_times = True

    """
    Classifications to give to each classification index
    Unfortunately, we need to assign these colors specific to classification, so we can't use the metadata from our samples.h5 archive
    """
    classifications = ["Healthy Tissue", "Type I - Caseum", "Type II", "Empty Slide", "Type III", "Type I - Rim", "Unknown/Other"]

    """
    BGR Colors to give to each classification index
              Pink,          Red,         Green,       Light Grey,      Yellow,        Blue         Purple
    """
    colors = [(255, 0, 255), (0, 0, 255), (0, 255, 0), (200, 200, 200), (0, 255, 255), (255, 0, 0), (244,66,143)]

    """
    Write our classification - color matchup metadata for future use
    """
    with open(classification_metadata_dir, "w") as f:
        pickle.dump(([classifications, colors]), f)

    """
    Load all our bounding rects
    """
    with open(rects_archive_dir, "r") as f:
        rects = pickle.load(f)

    """
    Open our saved models
        1 -> Type 1 classifications
        2 -> Type 2 & 3 classifications
    """
    classifier_1 = StaticConfig(model_1, model_dir)
    classifier_2 = StaticConfig(model_2, model_dir)

    """
    We open our image file, where each image is stored as a dataset with a key of it's index (e.g. '0', '1', ...)
    We also open our predictions file, where we will be writing our predictions in the same manner of our image file,
        so that they have a string index according to the image they came from.
    """
    with h5py.File(img_archive_dir, "r", chunks=True, compression="gzip") as img_hf:
        with h5py.File(predictions_archive_dir, "w") as predictions_hf:
            """
            Get image number for iteration
            """
            img_n = len(img_hf.keys())

            """
            We want to save time by not re-initializing a big block of memory for our image
                every time we get a new image, so we find the maximum shape an image will be here.
            We do this by the product since blocks of memory are not in image format, their size 
                only matters in terms of raw number.
            We do the max shape because we initialize the img block of memory, and then initialize a bunch of other
                blocks of memory around this in the loop. 
            This means if we had less than the max, and then encountered a larger img than our current allocated block of memory,
                it would have to reallocate. 

            As my friend explains: "[Let's say] you initialize to the first image. Then you allocate more memory for other things inside the loop. That likely "surrounds" the first image with other stuff. Then you need a bigger image. Python can't realloc the image in place, so it has to allocate a new version, copy all the data, then release the old one."
            So we instead initialize it to the maximum size an image will be.
            """
            max_shape = img_hf.get("0").shape
            for img_i in range(img_n):
                img_shape = img_hf.get(str(img_i)).shape
                if np.prod(img_shape) > np.prod(max_shape):
                    max_shape = img_shape

            """
            Now initialize our img block of memory to to this max_shape, 
                with the same data type as our images
            """
            img = np.zeros(max_shape, img_hf.get("0").dtype)

            """
            Start looping through images
            """
            for img_i in range(img_n):
                """
                Start our timer for this image
                """
                start_time = time.time()

                """
                Get our image by resizing our block of memory instead of deleting and recreating it,
                    and make sure to read directly from the dataset into this block of memory,
                    so that we can reference it quickly when iterating through subsections for classification later.
                """
                img_dataset = img_hf.get(str(img_i))
                img.resize(img_dataset.shape)
                img_dataset.read_direct(img)

                """
                Get these for easy reference later, now that our image's dimensions aren't changing
                """
                img_h = img.shape[0]
                img_w = img.shape[1]
                prediction_h = img_h // sub_h
                prediction_w = img_w // sub_w
                prediction_n = float(prediction_h*prediction_w)#So we don't compute this every time we want to print progress

                """
                Generate matrix to store predictions as we loop through our subsections, which we can later reshape to a 3-tensor.
                    This will become a 3 tensor because for any entry at index i, j, we have our vector of size class_n for the model's output probabilities .
                    This is opposed to just having an argmaxed index, where the element at index i, j would just be an integer.
                """
                predictions = np.zeros((prediction_h*prediction_w, len(classifications)))

                """
                Get our bounding rectangles for this image
                """
                img_detected_bounding_rects = rects[img_i]

                """
                We only do any of this if we actually have some bounding rectangles on our image,
                    so we check for that here.

                (It also will break if we don't have this check,
                    because when doing img_detected_bounding_rects[:,0] there won't be any elements in the array to index, 
                    since img_detected_bounding_rects = [])
                """
                if len(img_detected_bounding_rects) > 0:
                    """
                    We want our coordinates to be relative to subsection size instead of pixels, so we divide accordingly
                        and convert back to int again.
                    """
                    img_detected_bounding_rects[:,0] /= sub_w
                    img_detected_bounding_rects[:,1] /= sub_h
                    img_detected_bounding_rects[:,2] /= sub_w
                    img_detected_bounding_rects[:,3] /= sub_h
                    img_detected_bounding_rects = img_detected_bounding_rects.astype(np.uint16)

                """
                We then want a quick reference list with booleans, so that we can check if a given subsection index is 
                    inside a bounding rectangle in constant time, without needing to loop through our bounding rects.
                Also, since our bounding rects are currently 2d, we need to convert them to 1d. 

                We can do both of these at the same time by creating a np array of booleans (initalized to false) of shape (prediction_h, prediction_w),
                    then looping through all bounding rect pairs to set all elements our matrix inside the 2d rectangles to True,
                    then flattening this boolean matrix into a vector.

                This will give us our quick reference in the shape of a matrix, then flatten it to a vector for easy reference.
                """
                index_to_classifier_reference = np.zeros((prediction_h, prediction_w), dtype=bool)
                for rect_2d in img_detected_bounding_rects:
                    x1 = rect_2d[0]
                    y1 = rect_2d[1]
                    x2 = rect_2d[2]
                    y2 = rect_2d[3]
                    index_to_classifier_reference[y1:y2, x1:x2] = True

                index_to_classifier_reference = index_to_classifier_reference.flatten()

                """
                We then want a list of (i, row_i, col_i), as if we were iterating through subsections in our image.
                    For example, with sub_h = 80 and sub_w = 145, on a small image, we want the elements in this list to be 
                        [(0, 0, 0), (1, 0, 145), (2, 0, 290), (3, 80, 0), (4, 80, 145), (5, 80, 290), (6, 160, 0), ...]
                    Exactly if we were looping through like this (notes on why we loop on this below)

                        i = 0
                        for row_i in xrange(0, prediction_h*sub_h, sub_h):
                            for col_i in xrange(0, prediction_w*sub_w, sub_w):
                                yield (i, row_i, col_i)
                                i += 1

                        The important note here is the //, an integer division instead of a normal division. By doing this,
                            we make sure we only loop through the subsections where row_i + sub_h is complete, not going
                            outside the borders of our image.
                        The key difference is that (img.shape[0]//sub_h)*sub_h is different from img.shape[0] only because
                            the former no longer has any remainder when dividing it by sub_h, whereas img.shape[0] probably does.
                        This is important because when there is a remainder, we would return that partial subsection from this generator,
                            and that's not something we want to do because partials end up confusing our classifier.
                        Of course, this is the same reasoning for sub_w.

                    This is the exact code we would use if we wanted to iterate through this list, but since it's just integers
                        and we can create it with numpy, it's using up very little memory and we can create it really quickly.
                
                We will use the first element with our index to classifier reference to find which list to append each tuple to,
                    and we will use row_i and col_i (once they've been appended) to reference subsections in the original image.
                """
                i = np.arange(prediction_h*prediction_w)
                row_is, col_is = np.mgrid[0:prediction_h*sub_h:sub_h, 0:prediction_w*sub_w:sub_w]

                """
                We then reshape all of them into matrices of shape (prediction_h*prediction_w, 1) so we can concatenate them together.
                """
                i = np.reshape(i, (-1, 1))
                row_is = np.reshape(row_is, (-1, 1))
                col_is = np.reshape(col_is, (-1, 1))

                """
                And finally concatenate into one big tuple of each subsections' reference data for creating our classifier-specific input lists
                """
                sub_references = np.concatenate((i, row_is, col_is), axis=1)

                """
                Our classifier specific input lists, to be appended to as we loop through our subsection references
                """
                classifier_1_sub_references = []
                classifier_2_sub_references = []

                """
                Then loop through and use our index-to-classifier reference for appending each subsection reference to the appropriate classifier-specific list.
                """
                for sub_reference in sub_references:
                    """
                    Use the index already present and ordered in the reference tuple for checking since it should be the same constant complexity 
                        as if we kept track of the index with this loop.
                    """
                    if index_to_classifier_reference[sub_reference[0]]:
                        classifier_1_sub_references.append(sub_reference)
                    else:
                        classifier_2_sub_references.append(sub_reference)

                """
                Don't need this anymore
                """
                del sub_references

                """
                Cast our lists to arrays since they are not going to be changing in size anymore
                """
                classifier_1_sub_references = np.array(classifier_1_sub_references)
                classifier_2_sub_references = np.array(classifier_2_sub_references)

                """
                Create arrays for each classifier's predictions using the size of our new np arrays
                """
                classifier_1_predictions = np.zeros((classifier_1_sub_references.shape[0], len(classifications)))
                classifier_2_predictions = np.zeros((classifier_2_sub_references.shape[0], len(classifications)))

                """
                Since this is interacting with a file which has its own progress indicator,
                    we write some blank space to clear the screen of any previous text before writing any of our progress indicator
                """
                sys.stdout.write("\r                                       ")

                """
                We now loop through the sub references for each classifier by mini batch size, 
                    storing the results in our classifier's predictions with the batch.shape[0] 
                    for proper referencing - since we will have a different batch shape 
                    on the last iteration if our sub_references size for this classifier is not 
                    divisible (with no remainder) by our mini batch size
                """
                for sub_i in range(0, classifier_1_sub_references.shape[0], mb_n):
                    """
                    Print progress, using fancy formatting to avoid multiplying by 100 each time to print a percentage
                    """
                    sys.stdout.write("\rImage {} -> {:.2%} Complete".format(img_i, sub_i/(prediction_n)))
                    sys.stdout.flush()

                    """
                    Get the coordinates for referencing our image from our sub_references
                    """
                    img_references = classifier_1_sub_references[sub_i:sub_i+mb_n, 1:]

                    """
                    Create a np array of subs by referencing the correct subsection in our img with each img_reference in our img_references
                    """
                    classifier_1_subs = np.array([img[img_reference[0]:img_reference[0]+sub_h, img_reference[1]:img_reference[1]+sub_w] for img_reference in img_references], copy=False)

                    """
                    Get our predictions for our subsections
                    """
                    classifier_1_predictions = classifier_1.classify(classifier_1_subs)

                    """
                    Place these predictions in final predictions array, using our original index for each input
                        to know where to place them.
                    """
                    for classifier_1_prediction_i, classifier_1_prediction in enumerate(classifier_1_predictions):
                        """
                        We get the original (i, row_i, col_i) element with this:
                            classifier_1_sub_references[sub_i + classifier_1_prediction_i]
                        And then we get just the i element with this:
                            classifier_1_sub_references[sub_i + classifier_1_prediction_i][0]
                        And then we use that to reference predictions with this:
                            prediction_i = classifier_1_sub_references[sub_i + classifier_1_prediction_i][0]
                            predictions[prediction_i]
                        And then since our output vector needs to be mapped to length 7 from length 4, 
                            we assign in a unique way for this classifier.
                        """
                        prediction_i = classifier_1_sub_references[sub_i+classifier_1_prediction_i, 0]
                        predictions[prediction_i, 0:2] = classifier_1_prediction[0:2]
                        predictions[prediction_i, 3] = classifier_1_prediction[2]
                        predictions[prediction_i, 5] = classifier_1_prediction[3]

                """
                Now we do the same for the other classifier
                """
                for sub_i in range(0, classifier_2_sub_references.shape[0], mb_n):
                    """
                    Print progress, using fancy formatting to avoid multiplying by 100 each time to print a percentage. 
                    We also add the # of classifier_1_sub_references to get the correct total percentage progress, continued from our previous loop.
                    """
                    sys.stdout.write("\rImage {} -> {:.2%} Complete".format(img_i, (sub_i+classifier_1_sub_references.shape[0])/(prediction_n)))
                    sys.stdout.flush()

                    """
                    Get the coordinates for referencing our image from our sub_references
                    """
                    img_references = classifier_2_sub_references[sub_i:sub_i+mb_n, 1:]

                    """
                    Create a np array of subs by referencing the correct subsection in our img with each img_reference in our img_references
                    """
                    classifier_2_subs = np.array([img[img_reference[0]:img_reference[0]+sub_h, img_reference[1]:img_reference[1]+sub_w] for img_reference in img_references], copy=False)

                    """
                    Get our predictions for our subsections
                    """
                    classifier_2_predictions = classifier_2.classify(classifier_2_subs)

                    """
                    Place these predictions in final predictions array, using our original index for each input
                        to know where to place them.
                    """
                    for classifier_2_prediction_i, classifier_2_prediction in enumerate(classifier_2_predictions):
                        """
                        We get the original (i, row_i, col_i) element with this:
                            classifier_2_sub_references[sub_i + classifier_2_prediction_i]
                        And then we get just the i element with this:
                            classifier_2_sub_references[sub_i + classifier_2_prediction_i][0]
                        And then we use that to reference predictions with this:
                            prediction_i = classifier_2_sub_references[sub_i + classifier_2_prediction_i][0]
                            predictions[prediction_i]
                        And then since our output vector needs to be mapped to length 7 from length 4, 
                            we assign in a unique way for this classifier.
                        """
                        prediction_i = classifier_2_sub_references[sub_i+classifier_2_prediction_i, 0]
                        predictions[prediction_i, 0] = classifier_2_prediction[0]
                        predictions[prediction_i, 2:5] = classifier_2_prediction[1:]

                """
                Convert our now-complete predictions matrix into a 3-tensor so we can then store it into our dataset.
                """
                predictions = np.reshape(predictions, (img_h//sub_h, img_w//sub_w, len(classifications)))

                """
                TEMP
                We then denoise our predictions, now that all the predictions are loaded for this image.
                """
                predictions = post_processing.denoise_predictions(predictions, .8)

                """
                Then we store it in our dataset
                """
                predictions_hf.create_dataset(str(img_i), data=predictions)

                """
                Print how long this took, if you want to show off.
                """
                end_time = time.time() - start_time
                if print_times:
                    #\n added to ensure proper print formatting with sys.stdout between loops
                    print "\nTook %f seconds (%f minutes) to execute on image %i." % (end_time, end_time/60.0, img_i)
                else:
                    #Ensure proper print formatting with sys.stdout between loops
                    print ""


