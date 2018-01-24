import sys

import numpy as np
from keras.models import load_model

from base import *
from EditingDataset import *

class PredictionGrids(object):
    def __init__(self, dataset, uid):
        self.dataset = dataset#for reference, do not modify
        self.uid = uid

        #Our two attributes for predictions before and after editing.
        self.archive_dir_before_editing = "../data/prediction_grids_before_editing/"#Where we'll store the .npy files for our predictions before editing
        self.archive_dir_after_editing = "../data/prediction_grids_after_editing/"#Where we'll store the .npy files for our predictions after editing
        self.before_editing = EditingDataset(self.dataset, self.uid, self.archive_dir_before_editing)
        self.after_editing = EditingDataset(self.dataset, self.uid, self.archive_dir_after_editing)

        #Parameters
        self.class_n = 7
        self.sub_h = 80
        self.sub_w = 145

        #Classifiers
        self.type_one_classifier = load_model("../classifiers/type_one_classifier.h5")#For type-one classification
        self.non_type_one_classifier = load_model("../classifiers/non_type_one_classifier.h5")#For non-type-one classification

    def generate(self):
        """
        Loops through a grid of subsections in our image
            as input to our models,
        and outputs a grid of predictions matching these subsections.
        """
        
        #Generate for each image
        for img_i, img in enumerate(self.dataset.imgs):

            #Total # of predictions on this image
            prediction_h = (img.shape[0]//self.sub_h)
            prediction_w = (img.shape[1]//self.sub_w)
            prediction_n = (img.shape[0]//self.sub_h)*(img.shape[1]//self.sub_w)
            
            #Where predictions are stored for image. Starts as 1d for easier reference
            prediction_grid = np.zeros((prediction_n, self.class_n), dtype=np.uint8)
            
            #For knowing which classifier to use for a given input. Starts as 2d for easier reference
            classifier_reference = np.zeros((prediction_h, prediction_w), dtype=bool)

            """
            Build this reference by rescaling our image detections to match the image's subsection grid (and casting to int),
                then setting all entries in our classifier reference bounded by each detection
                to be True representing the inputs which are within detections.
            Since this won't work and doesn't make since if we don't have any detections, we check for that also before doing this.
            """
            detections = self.dataset.type_one_detections.after_editing[img_i]

            if len(detections) > 0:
                detections[:,0] = detections[:,0] / self.sub_w#x1
                detections[:,1] = detections[:,1] / self.sub_h#y1
                detections[:,2] = detections[:,2] / self.sub_w#x2
                detections[:,3] = detections[:,3] / self.sub_h#y2
                detections = detections.astype(np.uint16)
                for i, detection in enumerate(detections):
                    classifier_reference[detection[1]:detection[3], detection[0]:detection[2]] = True
            continue

            #Then reshape back to 1d so we can check each subsection against it easily
            classifier_reference = np.reshape(classifier_reference, (prediction_n,))

            #Loop through subsection inputs in image.
            for i, subsection in enumerate(subsections(img, self.sub_h, self.sub_w)):
                sys.stdout.write("\rGenerating Prediction Grid on Image %i/%i. %.2f%% Complete." % (img_i, len(self.dataset.imgs)-1, (i/prediction_n)*100.0))
                #Use our classifier reference vector to check which classifier this input belongs to
                if classifier_reference[i]:
                    #Get prediction for this classification with the correct classifier
                    prediction = self.type_one_classifier.predict(np.array([subsection]))[0]

                    #Convert the local output classification enumeration of this classifier to the global ones and insert
                    prediction_grid[i,0:2] = prediction[0:2]
                    prediction_grid[i,3] = prediction[2]
                    prediction_grid[i,5] = prediction[3]
                else:
                    #Get prediction for this classification with the correct classifier
                    prediction = self.non_type_one_classifier.predict(np.array([subsection]))[0]

                    #Convert the local output classification enumeration of this classifier to the global ones and insert
                    prediction_grid[i,0] = prediction[0]
                    prediction_grid[i,2:5] = prediction[1:]

            #Reshape prediction grid to 2d now that we have all predictions, and save
            self.before_editing[img_i] = np.reshape(prediction_grid, (prediction_h, prediction_w, self.class_n))

            sys.stdout.flush()
            print("")







    def edit(self):
        pass
