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
            Get our image, and use our dimensions to determine how big our prediction array needs to be for this image
                After we pad it with necessary zeros so that we never have partial edge problems with the far edges
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
            It's far easier to handle this when we start predictions as a vector, and convert it 
                to a matrix at the end, once we have all the predictions for the entire image.

            So, we store some values to help us out when we do that reshape later.
            """
            predictions_h = (img_h/sub_h)
            predictions_w = (img_w/sub_w)
            predictions_v_n = predictions_h*predictions_w
            predictions = np.zeros(shape=(predictions_v_n))

            """
            So we keep track of where we left off with our last prediction.
            """
            prediction_i = 0

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
                        

                    #Use these to get the next sub-image of our main image
                    sub_img = get_next_subsection(row_i, col_i, img_h, img_w, sub_h, sub_w, img, img_divide_factor)
                    sub_img_h = sub_img.shape[0]
                    sub_img_w = sub_img.shape[1]

                    #Generate our overlay with the shape of our sub_img, resized down by our final resize factor so as to save memory
                    overlay = np.zeros(shape=(int(sub_img_h//resize_factor), int(sub_img_w//resize_factor), 3))

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
                        Then, we classify the new batch of examples,
                            and store it in our global vector of predictions.
                        """
                        predictions[prediction_i:prediction_i+batch.shape[0]] = nn_classifier.classify(batch)
                        prediction_i += batch.shape[0]
                        
                    if verbose:
                        print ""#flush formatting
                    
                    """
                    Convert our predictions for this subsection into a matrix so we can reference it to use in the overlay
                        Since we have the same number of predictions as we have in subs, we know our predictions are from (prediction_i-subs_h*subs_w) to (prediction_i)
                    """
                    overlay_predictions = np.reshape(predictions[(prediction_i-(subs_h*subs_w)):prediction_i], (sub_img_h//sub_h, sub_img_w//sub_w))

                    #Now that we have all our predictions, loop through and generate respective overlay rectangles
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
                    sub_img_h = sub_img.shape[0]
                    sub_img_w = sub_img.shape[1]

                    #Set to the same type as our overlay
                    sub_img = sub_img.astype(np.float64)

                    #Replace value with scalar at the end to have a 3d matrix
                    sub_img = sub_img.reshape(sub_img_h, sub_img_w, 1)

                    #Copy value over 2nd axis to get rgb representation
                    sub_img = np.repeat(sub_img, 3, axis=2)

                    #Transparency weight of our overlay, percentage it takes up
                    alpha = .3333333

                    #Add our overlay to the sub_img
                    cv2.addWeighted(overlay, alpha, sub_img, 1-alpha, 0, sub_img)

                    #Write our combined img
                    cv2.imwrite('%s/%i_%i_%i.jpg' % (results_dir, img_i, row_i, col_i), sub_img)

                    img_sub_i +=1

            """
            Now that our entire image is done, we reshape our predictions vector back into an appropriate matrix for the image,
                and store it in the dataset.
            """
            if verbose:
                print "\tSaving Predictions..."
            predictions = np.reshape(predictions, (predictions_h, predictions_w))
            predictions_hf.create_dataset(str(img_i), data=predictions)

            end_time = time.time() - start_time
            if print_times:
                if not verbose:
                    print ""
                print "Took %f seconds (%f minutes) to execute on image %i." % (end_time, end_time/60.0, img_i)
