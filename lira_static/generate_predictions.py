import sys, os, time
import numpy as np
import h5py, pickle
import cv2

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
            divides the images into smaller subsections so as not to classify the entire image at once,
        Then goes through the subsections of our image, and divides into our sub_hxsub_w subsections, 
        then classifies each of these,
            using our first model if a subsection is inside a bounding rectangle, 
            and using our second model if a subsection is not inside a bounding rectangle.
        Then, it stores the predictions into a matrix of integers, or concatenates them onto the pre-existing predictions from previous subsections.
        Once completed with all subsections, these predictions are combined to get one 2d matrix of predictions, which is written to `predictions_archive_dir`.

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
    """
    sub_h = 80
    sub_w = 145

    """
    Our factor to resize our image by.

    Set this to None to have a dynamic resize factor.
        the threshold is only relevant if it is None.
    """
    dynamic_resize_factor = True
    resize_factor = None

    """
    Enable this if you want to see how long it took to classify each image
        *cough* if you want to show off *cough*
    """
    print_times = False

    """
    Then convert the subs from grayscale to rgb, by repeating their last dimension 3 times.
        We do this because our model's first stage consists of pretrained models, 
        which were trained on rgb data originally and thus expect data with 3 channels.
    Only enable this if you are using those models.
    """
    grayscale_to_rgb = False

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
                Get our resize_factor variable from the dimensions of our image
                """
                if dynamic_resize_factor:
                    resize_factor = float(get_relative_factor(img_h, resize_factor))
                
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
                    img_detected_bounding_rects = img_detected_bounding_rects.astype(int)

                    """
                    We then convert the img_detected_bounding_rectangles from an array of shape (n, 4) to one of shape (n, 2)
                        by converting our 2d rectangle coordinates into sets of 1d coordinates, 
                        which we can use to reference all the elements in our flattened image which were originally in our 2d rectangles.

                    We do this using our function for it in img_handler.py, and further documentation for how it works
                        can be found there.
                    """
                    img_detected_bounding_rects = convert_2d_rects_to_1d_rects(img_detected_bounding_rects, img_w//sub_w)

                """
                In order to handle everything easily and correctly, we do the following:
                    Each subsection of our image we get an overlay_predictions 3-tensor of predictions, 
                        which we either insert into the correct row, or concatenate onto the row's pre-existing values
                """
                predictions = [np.array([]) for i in range(img_divide_factor)]

                """
                We also start a counter for our predictions, which we increment with each batch we classify.
                    Using this, we can check if a new batch of subsections is inside our bounding rectangles.
                This is only reset on every new image, so we call it img_prediction_i 
                """
                img_prediction_i = 0

                """
                Here we start looping through rows and columns in our image, 
                and divide our main image into smaller images to handle one at a time, and save memory.
                """
                img_sub_i = 0

                """
                Since this is interacting with a file which has its own progress indicator,
                    we write some blank space to clear the screen of any previous text before writing any of our progress indicator
                """
                sys.stdout.write("\r                                       ")

                for row_i in range(img_divide_factor):
                    for col_i in range(img_divide_factor):
                        sys.stdout.write("\rIMAGE %i, SUBSECTION %i" % (img_i, img_sub_i))
                        sys.stdout.flush()
                            
                        """
                        So we keep track of where we left off with our last prediction.
                        """
                        overlay_prediction_i = 0

                        """
                        Get the next sub-image of our main image
                        """
                        sub_img = get_next_subsection(row_i, col_i, img_h, img_w, sub_h, sub_w, img, img_divide_factor)
                        sub_img_h = sub_img.shape[0]
                        sub_img_w = sub_img.shape[1]

                        """
                        Generate our overlay using the shape of our sub_img, resized down by our final resize factor so as to save memory
                        """
                        overlay = np.zeros(shape=(int(sub_img_h//resize_factor), int(sub_img_w//resize_factor), 3))

                        """
                        Generate matrix to store predictions as we loop through our subsections, which we can later reshape to a 3-tensor.
                            This will become a 3 tensor because for any entry at index i, j, we have our vector of size class_n for the model's output probabilities 
                            This is opposed to just having an argmaxed index, where index i, j would just have an integer.
                        """
                        overlay_predictions = np.zeros(((sub_img_h//sub_h)*(sub_img_w//sub_w), len(classifications)))

                        """
                        From our sub_img, get a matrix of subsections in this sub_img.
                        """
                        subs = get_subsections(sub_h, sub_w, sub_img, rgb=rgb)

                        """
                        Convert this matrix of subsections to a vector of subsections, with each entry being a subsection.
                            This way, we can easily loop through them.
                        """
                        if rgb:
                            subs = np.reshape(subs, (-1, sub_h, sub_w, 3))
                        else:
                            subs = np.reshape(subs, (-1, sub_h, sub_w, 1))

                        """
                        Then convert the subs from grayscale to rgb, by repeating their last dimension 3 times.
                            We do this because our model's first stage consists of pretrained models, 
                            which were trained on rgb data originally and thus expect data with 3 channels.
                        Only enable this if you are using those models.
                        """
                        if grayscale_to_rgb:
                            subs = np.repeat(subs, [3], axis=3)

                        """
                        Loop through vector of subsections with step mb_n
                        """
                        for sub_i in xrange(0, len(subs), mb_n):

                            """
                            Get our batch by referencing the right location in our subs with sub_i and mb_n
                            Note: This does get any extra samples, even if len(subs) % mb_n != 0
                            """
                            batch = subs[sub_i:sub_i+mb_n]

                            """
                            We then check each element in our batch to see if it is inside a bounding rectangle in our image,
                                and if so, we classify it with our first (type 1) classifier,
                                if not, we classify it with our second (type 2 & 3) classifier.

                            In the most general case, we'd need to loop through each element and iteratively add them to a
                                "classifier 1" set, or a "classifier 2" set. 
                            However, for all but the most bizarre edge case, this won't happen. Here are the possible cases:
                                1. No samples in our batch are in a rectangle.
                                2. The first n samples of our batch are in a rectangle, the rest are not.
                                3. The last n samples of our batch are in a rectangle, the rest are not.
                                4. All the samples in our batch are in a rectangle.
                                5. Some of the samples in our batch are in a rectangle, but at least one section that is in a rectangle is surrounded by sections that are not in rectangles,
                                    or sections in other rectangles.
                            The first 4 cases will almost always happen in an image, however the 5th will only happen if 
                                our window size for our bounding rectangles is less than our mini batch size.
                            For this problem case, our mini batch size can't go higher than 100 due to memory constraints,
                                and our window size is 512 (because lower would be a time constraint).
                            So for this problem, we know there are only 4 cases. Given this information, 
                                we can design a more efficient and simpler algorithm.

                            Note: It's more efficient in how it gives us exactly two indices for referencing the samples in a rectangle,
                                instead of a list. This is because it allows us to send all of these samples at once to the classifier,
                                instead of one at a time with the other method. It's also simpler, in my opinion.

                            We loop through each element in the batch, 
                                and then loop through each pair of 1d coordinates for our bounding rectangles,
                            Until we find one that is inside one of our pairs. 
                            This is our first element that is inside our bounding rectangle.
                            We then search for an element after this which is outside of our bounding rectangle,
                                defaulting to `width` or `batch.shape[0]` if we don't find one.
                            This is our last element that is inside our bounding rectangle.
                            We use the first and last element (if they are not found, this will not occur)
                                to reference all elements in our batch which are in a bounding rectangle, 
                                and classify these with our first classifier, using the second classifier for all others.
                            We initialise first and last to the same value (batch.shape[0]) so that we don't reference any elements
                                if none are found to be inside our bounding rectangle.
                            """
                            first_bounding_rect_i = batch.shape[0]
                            first_bounding_rect_i_found = False
                            last_bounding_rect_i = batch.shape[0]
                            
                            """
                            We only do any of this if we actually have some bounding rectangles on our image,
                                so we check for that here.
                            """
                            if len(img_detected_bounding_rects) > 0:
                                """
                                We have our img_prediction_i which is incremented each batch, however
                                    due to the fact that our batches of subsections are part of larger subsections which the image is
                                    divided into, we have to find the global img_prediction_i by taking these into account.
                                    This is a bit complicated, so we use a separate function for it
                                """
                                #def get_global_prediction_i(i, img_h, img_w, sub_h, sub_w, img_divide_factor, sub_img_row_i, sub_img_col_i):
                                #def get_global_prediction_i(i, img_h, img_w, sub_h, sub_w, img_divide_factor, sub_img_row_i, sub_img_col_i):
                                """
                                print img_prediction_i, img_h, img_w, sub_h, sub_w, img_divide_factor, row_i, col_i
                                sys.exit()
                                """

                                global_prediction_i = get_global_prediction_i(img_prediction_i, img_h, img_w, sub_h, sub_w, img_divide_factor, row_i, col_i)
                                #print img_prediction_i, global_prediction_i

                                """
                                First loop, to search for first_bounding_rect_i

                                Note: We have to add our img_prediction_i to sample_i in order to get
                                    our sample's position in the entire image, in order to compare it with our bounding rectangles,
                                    which are coordinates over the entire image
                                    
                                    However, we keep our first_bounding_rect_i and last_bounding_rect_i as local (without this offset),
                                        in order to easily reference the elements in our batch using them.
                                """
                                for sample_i, sample in enumerate(batch):
                                    sample_i += global_prediction_i
                                    for pair in img_detected_bounding_rects:
                                        if pair[0] <= sample_i and sample_i <= pair[1]:
                                            """
                                            Our first bounding rect index, store without offset and break out of this loop
                                            """
                                            first_bounding_rect_i = sample_i - global_prediction_i
                                            first_bounding_rect_i_found = True
                                            break

                                    if first_bounding_rect_i_found:
                                        break

                                """
                                Second loop, to search for last_bounding_rect_i
                                    (only happens if we find first_bounding_rect_i)
                                We know we've found the element when it's not inside any of our rectangles.
                                If we don't find it, we default to our initial value, batch.shape[0]

                                Note: We have to add our img_prediction_i to sample_i in order to get
                                    our sample's position in the entire image, in order to compare it with our bounding rectangles,
                                    which are coordinates over the entire image
                                    
                                    However, we keep our first_bounding_rect_i and last_bounding_rect_i as local (without this offset),
                                        in order to easily reference the elements in our batch using them.
                                """
                                if first_bounding_rect_i_found:
                                    for sample_i, sample in enumerate(batch[first_bounding_rect_i:]):
                                        sample_i += (global_prediction_i + first_bounding_rect_i)
                                        for pair in img_detected_bounding_rects:
                                            if pair[0] <= sample_i and sample_i <= pair[1]:
                                                break
                                        else:
                                            """
                                            Our last bounding rect index, store without offset and break out of this loop
                                            """
                                            last_bounding_rect_i = sample_i - global_prediction_i
                                            break

                            """
                            So at this point we have two indices for our bounding rectangle elements.
                            We can use these to reference the correct elements of our batch for each classifier.

                            Since our classification indices are going to be 0, 1, 2, 3 for both first and second classifiers, 
                                however they represent different things for each of them
                                (e.g. 1 = Type 1 Caseum for 1st classifier, 1 = Type II for 2nd classifier),
                                which means we have to make sure they have unique and different numbers again = going from local to global mappings again.

                            In order to do this, we just manually map them from their unique mappings back to global, so that
                                1 = Type 1 Caseum 
                                2 = Type II 
                            Since these are probability vectors instead of indices, this is a bit more complicated, but overrall the same.
                            And we can then properly generate our overlays with our overlay_predictions array when we're done
                            """
                            #print first_bounding_rect_i, last_bounding_rect_i
                            if first_bounding_rect_i > 0:
                                """
                                If our first bounding rect index is not the first index, we classify all the samples up to it with our second classifier
                                    and insert them in the correct order for our mapping
                                """
                                #classifier_2_classifications = classifier_2.classify(batch[:first_bounding_rect_i])
                                classifier_2_classifications = np.array([[0,.9,.1,0]])#TEMP

                                overlay_predictions[overlay_prediction_i:overlay_prediction_i+first_bounding_rect_i][:,0] = classifier_2_classifications[:,0]
                                overlay_predictions[overlay_prediction_i:overlay_prediction_i+first_bounding_rect_i][:,2:5] = classifier_2_classifications[:,1:]
                                #print "TYPE 2"

                            if last_bounding_rect_i > first_bounding_rect_i:
                                """
                                If we have some elements in the rectangle, we classify all of them with our first classifier
                                    and insert them in the correct order for our mapping
                                """
                                #classifier_1_classifications = classifier_1.classify(batch[first_bounding_rect_i:last_bounding_rect_i])
                                classifier_1_classifications = np.array([[0,.9,.1,0]])#TEMP

                                overlay_predictions[overlay_prediction_i+first_bounding_rect_i:overlay_prediction_i+last_bounding_rect_i][:,0:2] = classifier_1_classifications[:,0:2]
                                overlay_predictions[overlay_prediction_i+first_bounding_rect_i:overlay_prediction_i+last_bounding_rect_i][:,3] = classifier_1_classifications[:,2]
                                overlay_predictions[overlay_prediction_i+first_bounding_rect_i:overlay_prediction_i+last_bounding_rect_i][:,5] = classifier_1_classifications[:,3]
                                #print "TYPE 1 ", overlay_predictions[overlay_prediction_i:overlay_prediction_i+last_bounding_rect_i]
                                #print "TYPE 1"

                            if last_bounding_rect_i < batch.shape[0]-1:
                                """
                                If we have elements after our elements in the rectangle, we classify all of them with our second classifer
                                    and insert them in the correct order for our mapping
                                """
                                #overlay_predictions[overlay_prediction_i+last_bounding_rect_i:] = classifier_2.classify(batch[last_bounding_rect_i:])
                                #classifier_2_classifications = classifier_2.classify(batch[last_bounding_rect_i:])
                                classifier_2_classifications = np.array([[0,.9,.1,0]])#TEMP

                                overlay_predictions[overlay_prediction_i+last_bounding_rect_i:overlay_prediction_i+batch.shape[0]][:,0] = classifier_2_classifications[:,0]
                                overlay_predictions[overlay_prediction_i+last_bounding_rect_i:overlay_prediction_i+batch.shape[0]][:,2:5] = classifier_2_classifications[:,1:]

                                #print "TYPE 2 ", overlay_predictions[overlay_prediction_i:overlay_prediction_i+last_bounding_rect_i]
                                #print "TYPE 2"

                            """
                            print first_bounding_rect_i, last_bounding_rect_i
                            print overlay_predictions[overlay_prediction_i:overlay_prediction_i+batch.shape[0]]

                            if overlay_prediction_i > 12:
                                sys.exit()
                            """
                            """
                            We then increment our counters now that we have our new classifications stored
                            """
                            overlay_prediction_i += batch.shape[0]
                            img_prediction_i += batch.shape[0]
                            
                        """
                        Convert our predictions for this subsection into a 3-tensor so we can then concatenate it easily into our final predictions 3-tensor.
                        """
                        overlay_predictions = np.reshape(overlay_predictions, (sub_img_h//sub_h, sub_img_w//sub_w, len(classifications)))

                        """
                        Move our overlay_predictions matrix into our final predictions matrix, in the correct row according to row_i.
                            If this is the first column in a row:
                                Insert into row_i
                            If this is not, concatenate onto the correct row.
                        """
                        if col_i == 0:
                            predictions[row_i] = overlay_predictions
                        else:
                            predictions[row_i] = get_concatenated_row((predictions[row_i], overlay_predictions))

                        """
                        Increment total subsection number for display
                        """
                        img_sub_i +=1

                """
                Now that our entire image is done and our rows of concatenated predictions are ready,
                    we combine all the rows into a final matrix by concatenating into a column.
                """
                predictions = get_concatenated_col(predictions)

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
