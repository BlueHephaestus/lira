import sys, time
import numpy as np
#import gzip, cPickle, pickle
import h5py
import cv2

import static_config
from static_config import StaticConfig

import img_divider
from img_divider import *

#Directories
nn = "current_best_setup"
nn_dir = "../lira/lira1/src"
archive_dir = "../lira/lira1/data/greyscales.h5"
results_dir = "results_4x4"
#classification_metadata_dir = "classification_metadata.pkl"

#Parameters
mb_n = 100
img_divide_factor = 4
resize_factor = 16
sub_h = 80
sub_w = 145

resize_factor = float(resize_factor)

#Classifications to give to each classification index
#classifications = ["Type III", "Healthy Tissue", "Empty Slide", "Type I - Caseum", "Type II", "Type I - Rim", "Confidence < 50%"]
with h5py.File(img

#BGR Colors to give to each classification index
#         Yellow,        Pink,          White(mostly),   Red,         Green,       Blue,        Black
colors = [(0, 255, 255), (255, 0, 255), (200, 200, 200), (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 0, 0)]

#Write our classification - color matchup metadata for future use
f = open(classification_metadata_dir, "w")
pickle.dump(([classifications, colors]), f)

#Initialise net
nn_classifier = StaticConfig(nn, nn_dir)

#Open our greyscales so we can loop through
#f = gzip.open(img_archive_dir, 'rb')

#To keep up with the number for when we save the results
img_i = 0
#try:
while True:
    """
    Every time we load this we get a new greyscale, 
    So in order to keep going without knowing how many greyscales
    we have, we just keep trying to load until it fails, at which
    point we are done and end.
    """
    img = cPickle.load(f)

    #Divide into smaller images to handle one at a time
    #Given image and factor, returns vector with divided elements
    #img = (img, img_divide_factor)
    img_sub_i = 0

    for row_i in range(img_divide_factor):
        for col_i in range(img_divide_factor):

            print "IMAGE %i, SUBSECTION %i" % (img_i, img_sub_i)
            sys.stdout.write("\r\tGetting Subsections")
            #Use these to get the next sub-image of our main image
            sub_img = get_next_subsection(row_i, col_i, img.shape[1]//img_divide_factor, img.shape[0]//img_divide_factor, img)

            #Generate our overlay with the shape of our sub_img, resized down by our final resize factor so as to save memory
            overlay = np.zeros(shape=(int(sub_img.shape[0]//resize_factor), int(sub_img.shape[1]//resize_factor), 3))

            #Divide sub_img into subsections matrix to classify one at a time
            subs = get_subsections(sub_w, sub_h, sub_img)

            #So we don't have to compute these every time for percent completion calculation
            rows = float(len(subs))
            cols = float(len(subs[0]))
            """
            Percent completion relative to row index and column index
                is just row_i/len(rows) + col_i/(len(rows) * len(cols))
            This will not yield 100%, just extremely close.
            """
            perc_completion = lambda row_i, col_i: (row_i/rows + col_i/(rows * cols)) * 100

            #First, convert to vector of subsections, with each cell being the vectorized subsection
            subs = np.reshape(subs, (-1, sub_h, sub_w, 1))

            #Make new vector for predictions of each subsection in subsection vector
            predictions = np.zeros(shape=(len(subs)))

            #Loop through vector of subsections with step mb_n
            for sub_i in xrange(0, len(subs), mb_n):

                #Print percent completion for progress
                perc_complete = sub_i/float(len(subs)) * 100
                sys.stdout.write("\r\tClassifying Subsections - %02f%%" % (perc_complete))

                """
                Note: This method also gets any extras, if len(subs) % mb_n != 0

                Get our batch and assign resulting predictions
                    We first normalize by our previously used data normalization mean and standard deviation,
                    since we are obtaining unnormalized data from the image we normalize before feeding in the batch for prediction
                """
                batch = (subs[sub_i:sub_i+mb_n]*nn_classifier.stddev) + nn_classifier.mean

                """
                Then, we classify the new batch of examples.
                """
                predictions[sub_i:sub_i+mb_n] = nn_classifier.classify(batch)

            print " - Complete."

            #Reshape predictions to matrix of same dimensions as original subs matrix
            predictions = np.reshape(predictions, (sub_img.shape[0]//sub_h, sub_img.shape[1]//sub_w))

            #Now that we have all our predictions, loop through and generate respective overlay rectangles
            sys.stdout.write("\r\tGenerating Overlay")
            for prediction_row_i, prediction_row in enumerate(predictions):
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

            print " - Complete."
            sys.stdout.write("\r\tResizing and Saving Result Image")

            """
            Resize to match the size of our overlay. I originally had this with the decrease-by-ratio method,
                so as to go with the ratio we already have, however it's possible for it to have one more or less than 
                our overlay due to integer division rounding, so this way we assure they match.
            """
            sub_img = cv2.resize(sub_img, (overlay.shape[1], overlay.shape[0]))

            #Set to the same type as our overlay
            sub_img = sub_img.astype(np.float64)

            #Replace value with scalar at the end to have a 3d matrix
            sub_img = sub_img.reshape(sub_img.shape[0], sub_img.shape[1], 1)

            #Copy value over 2nd axis to get rgb representation
            sub_img = np.repeat(sub_img, 3, axis=2)

            #Transparency weight of our overlay, percentage it takes up
            alpha = .3333333

            #Add our overlay to the sub_img
            cv2.addWeighted(overlay, alpha, sub_img, 1 - alpha, 0, sub_img)

            #Write our combined img
            cv2.imwrite('%s/%i_%i_%i.jpg' % (results_dir, img_i, row_i, col_i), sub_img)

            img_sub_i +=1

            print " - Complete."

    #Increment img
    img_i += 1
    print "IMAGE NUMBER %i" % img_i

    #Go to next img
    #sys.exit()

"""
except:
    #No more greyscales, end
    pass
"""
