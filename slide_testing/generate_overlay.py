import sys, time
import numpy as np
import h5py, pickle
import cv2

import static_config
from static_config import StaticConfig

import img_handler
from img_handler import *

#Directories
nn = "current_best_setup"
nn_dir = "../lira/lira1/src"
img_archive_dir = "../lira/lira1/data/greyscales.h5"
predictions_archive_dir = "../lira/lira1/data/predictions.h5"
classification_metadata_dir = "classification_metadata.pkl"
results_dir = "results"

#Mostly static, not open for easy change yet.
sub_h = 80
sub_w = 145

#Parameters
mb_n = 100

"""
Set this to None to have a dynamic divide factor.
    the threshold is only relevant if it is None.

Same goes for resize factor.
"""
dynamic_img_divide_factor = True
img_divide_factor = None

dynamic_resize_factor = True
resize_factor = None

#Output options
verbose = False
print_times = False

#Classifications to give to each classification index
#Unfortunately, we need to assign these colors specific to classification, so we can't use the metadata from our samples.h5 archive
classifications = ["Healthy Tissue", "Type I - Caseum", "Type II", "Empty Slide", "Type III", "Type I - Rim"]

#BGR Colors to give to each classification index
#         Pink,          Red,         Green,       Light Grey,      Yellow,        Blue
colors = [(255, 0, 255), (0, 0, 255), (0, 255, 0), (200, 200, 200), (0, 255, 255), (255, 0, 0)]

#Clear our results directory of any pre-existing images
clear_dir(results_dir)

#Write our classification - color matchup metadata for future use
f = open(classification_metadata_dir, "w")
pickle.dump(([classifications, colors]), f)

#Initialise net
nn_classifier = StaticConfig(nn, nn_dir)

"""
We open our image file, where each image is stored as a dataset with a key of it's index (e.g. '0', '1', ...)
We also open our predictions file, where we will be writing our predictions in the same manner of our image file,
    so that they have a string index according to the image they came from.
"""
with h5py.File(img_archive_dir, "r") as img_hf:
    with h5py.File(predictions_archive_dir, "w") as predictions_hf:
        """
        We first get the number of images we have by counting the number of datasets,
        """
        img_n = len(img_hf.keys())

        for img_i in range(img_n):
            """
            Start our timer
            """
            start_time = time.time()

            """
            Get our image, and pad it with necessary zeros so that we never have partial edge problems with the far edges
            """
            img = np.array(img_hf.get(str(img_i)))
            img = pad_img(img.shape[0], img.shape[1], sub_h, sub_w, img)

            img_h = img.shape[0]
            img_w = img.shape[1]

            #Get our img_divide_factor and resize_factor variables
            if dynamic_img_divide_factor:
                img_divide_factor = get_relative_factor(img_h, img_divide_factor)
            if dynamic_resize_factor:
                resize_factor = float(get_relative_factor(img_h, resize_factor))
            
            """
            In order to handle everything easily and correctly, we do the following:
                Each subsection of our image we get an overlay_predictions matrix of predictions, 
                    which we either insert into the correct row, or concatenate onto the row's pre-existing values
            """
            predictions = [np.array([]) for i in range(img_divide_factor)]

            """
            Divide our main image into smaller images to handle one at a time, and save memory.
            """
            img_sub_i = 0
            for row_i in range(img_divide_factor):
                for col_i in range(img_divide_factor):

                    if verbose:
                        print "IMAGE %i, SUBSECTION %i" % (img_i, img_sub_i)
                        print "\tGetting Subsections..."
                    else:
                        sys.stdout.write("\rIMAGE %i, SUBSECTION %i" % (img_i, img_sub_i))
                        sys.stdout.flush()
                        
                    """
                    So we keep track of where we left off with our last prediction.
                    """
                    overlay_prediction_i = 0

                    #Use these to get the next sub-image of our main image
                    sub_img = get_next_subsection(row_i, col_i, img_h, img_w, sub_h, sub_w, img, img_divide_factor)
                    sub_img_h = sub_img.shape[0]
                    sub_img_w = sub_img.shape[1]

                    #Generate our overlay with the shape of our sub_img, resized down by our final resize factor so as to save memory
                    overlay = np.zeros(shape=(int(sub_img_h//resize_factor), int(sub_img_w//resize_factor), 3))

                    #Generate our predictions for the overlay as a vector at first, of size h * w = sub_img_h//sub_h * sub_img_w//sub_w . We later change this to a matrix.
                    overlay_predictions = np.zeros(((sub_img_h//sub_h)*(sub_img_w//sub_w)))

                    #Divide sub_img into subsections matrix to classify one at a time
                    subs = get_subsections(sub_h, sub_w, sub_img, verbose)

                    #So we don't have to compute these every time for percent completion calculation
                    subs_h = subs.shape[0]
                    subs_w = subs.shape[1]
                    """
                    Percent completion relative to row index and column index
                        is just row_i/len(rows) + col_i/(len(rows) * len(cols))
                    This will not yield 100%, just extremely close.
                    """
                    perc_completion = lambda row_i, col_i: (row_i/subs_h + col_i/(subs_h * subs_w)) * 100

                    #First, convert to vector of subsections, with each cell being the vectorized subsection
                    subs = np.reshape(subs, (-1, sub_h, sub_w, 1))

                    #Loop through vector of subsections with step mb_n
                    for sub_i in xrange(0, len(subs), mb_n):

                        #Print percent completion for progress
                        perc_complete = sub_i/float(len(subs)) * 100
                        if verbose:
                            sys.stdout.write("\r\tClassifying Subsections - %02f%%" % (perc_complete))
                            sys.stdout.flush()

                        """
                        Note: This method also gets any extras, if len(subs) % mb_n != 0

                        Get our batch and assign resulting predictions
                            We first normalize by our previously used data normalization mean and standard deviation,
                            since we are obtaining unnormalized data from the image we normalize before feeding in the batch for prediction
                        """
                        batch = (subs[sub_i:sub_i+mb_n]*nn_classifier.stddev) + nn_classifier.mean

                        """
                        Then, we classify the new batch of examples and store in our temporary predictions np.array
                        """
                        overlay_predictions[overlay_prediction_i:overlay_prediction_i+batch.shape[0]] = nn_classifier.classify(batch)
                        overlay_prediction_i += batch.shape[0]
                        
                    if verbose:
                        print ""#flush formatting
                    
                    """
                    Convert our predictions for this subsection into a matrix so we can reference it easily when making the overlay, 
                        and can then move it appropriately into our final predictions matrix.
                    """
                    overlay_predictions = np.reshape(overlay_predictions, (sub_img_h//sub_h, sub_img_w//sub_w))

                    """
                    Move our overlay_predictions matrix into our final predictions matrix, in the correct row according to row_i.
                        If this is the first column in a row:
                            Insert into row_i
                        If this is not, concatenate onto the correct row.
                    """
                    if col_i == 0:
                        predictions[row_i] = overlay_predictions
                    else:
                        predictions[row_i] = np.concatenate((predictions[row_i], overlay_predictions), axis=1)

                    #Now that we have all our predictions for this subsection, loop through and generate respective overlay rectangles
                    if verbose:
                        print "\tGenerating Overlay..."
                    for prediction_row_i, prediction_row in enumerate(overlay_predictions):
                        for prediction_col_i, prediction in enumerate(prediction_row):
                    
                            #Get the string classification and overlay color with our prediction index
                            prediction = int(prediction)
                            #classification = classifications[prediction]
                            color = colors[prediction]

                            """
                            Draw a rectangle with
                               the location specified by our indices, 
                               the size specified by our subsection size,
                               and our already known color.

                            We do int(...//resize_factor) so as to scale down the locations and sizes of each rectangle,
                               for our already resized overlay
                            """
                            cv2.rectangle(overlay, (int(prediction_col_i*sub_w//resize_factor), int(prediction_row_i*sub_h//resize_factor)), (int(prediction_col_i*sub_w+sub_w//resize_factor), int(prediction_row_i*sub_h+sub_h//resize_factor)), color, -1)

                    if verbose:
                        print "\tResizing and Saving Result Image..."

                    """
                    Resize to match the size of our overlay. I originally had this with the decrease-by-ratio method,
                        so as to go with the ratio we already have, however it's possible for it to have one more or less than 
                        our overlay due to integer division rounding, so this way we assure they match.
                    """
                    sub_img = cv2.resize(sub_img, (overlay.shape[1], overlay.shape[0]))

                    #Add our overlay
                    sub_img = add_weighted_overlay(sub_img, overlay, 0.3333333)

                    #Write our combined img
                    cv2.imwrite('%s/%i_%i_%i.jpg' % (results_dir, img_i, row_i, col_i), sub_img)

                    img_sub_i +=1

            if verbose:
                print "\tSaving Predictions..."

            """
            Now that our entire image is done and our rows of concatenated predictions are ready,
                we combine all the rows into a final matrix,
                and store it in the dataset.
            """
            predictions = np.concatenate([prediction_row for prediction_row in predictions], axis=0)

            predictions_hf.create_dataset(str(img_i), data=predictions)

            end_time = time.time() - start_time
            if print_times:
                if not verbose:
                    print ""
                print "Took %f seconds (%f minutes) to execute on image %i." % (end_time, end_time/60.0, img_i)
