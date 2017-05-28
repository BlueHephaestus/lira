import sys, os, time
import numpy as np
import h5py, pickle

import static_config
from static_config import StaticConfig

import img_handler
from img_handler import *

import object_detection_handler
from object_detection_handler import ObjectDetector

def generate_predictions(model_1, model_2, object_detection_model, model_dir,  img_archive_dir = "../lira/lira1/data/greyscales.h5", predictions_archive_dir = "../lira/lira1/data/predictions.h5", classification_metadata_dir = "classification_metadata.pkl", rgb=False):
    """
    Arguments:
        model_1: String containing the filename of where our first model is stored, to be used for classifying type 1 images and obtaining predictions
        model_2: String containing the filename of where our second model is stored, to be used for classifying type 2 and 3 images and obtaining predictions
        object_detection_model: String containing the filename of where our detection model is stored, to be used for detecting Type 1 Classifications in our images, 
            and drawing bounding rectangles on them.
        model_dir: Directory of where all our models are stored. Will be used with `model_1`, `model_2`, and `object_detection_model` to obtain where the model is now located 
        img_archive_dir: a string filepath of the .h5 file where the images / greyscales are stored.
        predictions_archive_dir: a string filepath of the .h5 file where the model's predictions on the images/greyscales will be stored.
        classification_metadata_dir: a string filepath of the .pkl file where the model's classification strings and color key for each classification will be stored.
        rgb: Boolean for if we are handling rgb images (True), or grayscale images (False).

    Returns:
        Goes through each image, 
            Gets bounding rectangles on any detected Type 1 classifications,
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
    f = open(classification_metadata_dir, "w")
    pickle.dump(([classifications, colors]), f)

    """
    Open our saved models
        1 -> Type 1 classifications
        2 -> Type 2 & 3 classifications
    """
    classifier_1 = StaticConfig(model_1, model_dir)
    classifier_2 = StaticConfig(model_2, model_dir)

    """
    Open our saved object detection model
    """
    object_detector = ObjectDetector(object_detection_model, model_dir)

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
            Start looping through images
            """
            for img_i in range(img_n):
                """
                Start our timer for this image
                """
                start_time = time.time()

                """
                Get our image
                """
                img = img_hf.get(str(img_i))

                """
                Get these for easy reference later, now that our image's dimensions aren't changing
                """
                img_h = img.shape[0]
                img_w = img.shape[1]

                """
                Get all bounding rectangles for type 1 classifications using our detection model, on our entire image.
                    (parameters for this model are defined in the object_detection_handler.py file, because this is a very problem-specific addition to this file)
                """
                img_detected_bounding_rects = object_detector.generate_bounding_rects(img)

                """
                We only do any of this if we actually have some bounding rectangles on our image,
                    so we check for that here.
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
                    img_detected_bounding_rects = img_detected_bounding_rects.astype(np.uint8)

                    """
                    We then convert the img_detected_bounding_rectangles from an array of shape (n, 4) to one of shape (n, 2)
                        by converting our 2d rectangle coordinates into sets of 1d coordinates, 
                        which we can use to reference all the elements in our flattened image which were originally in our 2d rectangles.

                    We do this using our function for it in img_handler.py, and further documentation for how it works
                        can be found there.
                    """
                    img_detected_bounding_rects = convert_2d_rects_to_1d_rects(img_detected_bounding_rects, img_w//sub_w)

                """
                Since this is interacting with a file which has its own progress indicator,
                    we write some blank space to clear the screen of any previous text before writing any of our progress indicator
                """
                sys.stdout.write("\r                                       ")
                """
                sys.stdout.write("\rIMAGE %i" % (img_i))
                sys.stdout.flush()
                """

                """
                Generate matrix to store predictions as we loop through our subsections, which we can later reshape to a 3-tensor.
                    This will become a 3 tensor because for any entry at index i, j, we have our vector of size class_n for the model's output probabilities .
                    This is opposed to just having an argmaxed index, where the element at index i, j would just be an integer.
                """
                predictions = np.zeros(((img_h//sub_h)*(img_w//sub_w), len(classifications)))

                """
                Loop through all of our individual subsections easily using a generator
                """
                for sub_i, sub in enumerate(subsections_generator(img, sub_h, sub_w)):
                    sys.stdout.write("\rSubsection %.2f" % (float(sub_i)/((img_h//sub_h)*(img_w//sub_w))))
                    sys.stdout.flush()

                    """
                    We then check each subsection to see if it is inside a bounding rectangle in our image,
                        and if so, we classify it with our first (type 1) classifier,
                        if not, we classify it with our second (type 2 & 3) classifier.
                    We use this flag for recording if it is inside a bounding rectangle, as you can probably tell.
                    """
                    sub_is_inside_bounding_rect = False

                    """
                    We only do any of this if we actually have some bounding rectangles on our image,
                        so we check for that here.
                    """
                    if len(img_detected_bounding_rects) > 0:
                        """
                        We loop through each of the pairs of bounding points, 
                            simply checking if our sub_i is inside any.
                        If we find it is, we know it is inside of a bounding rect,
                            so we set the flag and stop checking.
                        """
                        for pair in img_detected_bounding_rects:
                            if pair[0] <= sub_i and sub_i <= pair[1]:
                                sub_is_inside_bounding_rect = True
                                break
                        
                    """
                    Our classification indices are going to be 0, 1, 2, 3 for both first and second classifiers, 
                        however they represent different things for each of them;
                        (e.g. 1 = Type 1 Caseum for 1st classifier, 1 = Type II for 2nd classifier),
                        which means we have to make sure they have unique and different numbers again, we are going from local to global mappings again.

                    In order to do this, we just manually map them from their unique mappings back to global, so that
                        1 = Type I Caseum 
                        2 = Type II 
                        (for example)
                    Since each entry here is a probability vector instead of indices, this is a bit more complicated, but overrall the same.
                    When we're done, we can then properly generate our overlays with our predictions array
                    """
                    if sub_is_inside_bounding_rect:
                        """
                        If our subsection is inside a bounding rectangle, we classify it with our first classifier
                            and insert them in the correct order for our mapping
                        """
                        classifier_1_classification = classifier_1.classify(np.array([sub]))

                        predictions[sub_i,0:2] = classifier_1_classification[0,0:2]
                        predictions[sub_i,3] = classifier_1_classification[0,2]
                        predictions[sub_i,5] = classifier_1_classification[0,3]

                    else:
                        """
                        If our subsection is not inside a bounding rectangle, we classify it with our second classifier
                            and insert them in the correct order for our mapping
                        """
                        classifier_2_classification = classifier_2.classify(np.array([sub]))

                        predictions[sub_i,0] = classifier_2_classification[0,0]
                        predictions[sub_i,2:5] = classifier_2_classification[0,1:]


                """
                Convert our now-complete predictions matrix into a 3-tensor so we can then store it into our dataset.
                """
                predictions = np.reshape(predictions, (img_h//sub_h, img_w//sub_w, len(classifications)))

                """
                Then we store it in our dataset
                """
                predictions_hf.create_dataset(str(img_i), data=predictions)

                """
                Print how long this took, if you want to show off.
                """
                end_time = time.time() - start_time
                if print_times:
                    print "Took %f seconds (%f minutes) to execute on image %i." % (end_time, end_time/60.0, img_i)
