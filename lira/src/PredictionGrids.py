import sys

import numpy as np
from keras.models import load_model

from base import *
from EditingDataset import EditingDataset
from ImageSubsections import ImageSubsections
from PredictionGridEditor import PredictionGridEditor
from post_processing import denoise_predictions

class PredictionGrids(object):
    def __init__(self, dataset, uid, restart=False):
        self.dataset = dataset#for reference, do not modify
        self.uid = uid
        self.restart = restart

        #Our two attributes for predictions before and after editing.
        self.archive_dir_before_editing = "../data/prediction_grids_before_editing/"#Where we'll store the .npy files for our predictions before editing
        self.archive_dir_after_editing = "../data/prediction_grids_after_editing/"#Where we'll store the .npy files for our predictions after editing
        self.before_editing = EditingDataset(self.dataset, self.uid, self.archive_dir_before_editing, restart=self.restart)
        self.after_editing = EditingDataset(self.dataset, self.uid, self.archive_dir_after_editing, restart=self.restart)

        #Parameters
        self.class_n = 7
        self.sub_h = 80
        self.sub_w = 145
        self.mb_n = 24#Optimal Mini batch size
        self.denoising_weight = 0.8#Amount to denoise, 0 <= denoising_weight <= 1. Higher means more denoising.

        #Classifiers
        self.type_one_classifier = load_model("../classifiers/type_one_classifier.h5")#For type-one classification
        self.non_type_one_classifier = load_model("../classifiers/non_type_one_classifier.h5")#For non-type-one classification

    def generate(self):
        #Loops through a grid of subsections in our image as input to our models,
        #and outputs a grid of predictions matching these subsections.
        
        #Generate for each image
        for img_i, img in enumerate(self.dataset.imgs):

            #Total # of predictions on this image
            prediction_h = (img.shape[0]//self.sub_h)
            prediction_w = (img.shape[1]//self.sub_w)
            prediction_n = (img.shape[0]//self.sub_h)*(img.shape[1]//self.sub_w)
            
            #Where predictions are stored for image. Starts as 1d for easier reference
            prediction_grid = np.zeros((prediction_n, self.class_n), dtype=np.float32)
            
            #Class to interface with the image as if it were a vector of subsections of size self.sub_hxself.sub_w, without actually dividing the image as doing so would require too much storage.
            img_subsections = ImageSubsections(img, self.sub_h, self.sub_w)

            #For knowing which classifier to use for a given input. Starts as 2d for easier reference
            type_one_mask = np.zeros((prediction_h, prediction_w), dtype=bool)

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
                    type_one_mask[detection[1]:detection[3], detection[0]:detection[2]] = True

            #Then reshape back to 1d so we can easily use it as a mask
            type_one_mask = np.reshape(type_one_mask, (prediction_n,))

            #Get indices of all entries in type_one_mask (and therefore the indices of all subsections which are to be classified by our type-one classifier) 
            #   which are true. These are the subsections as input for the type-one classifier
            type_one_subsection_indices = np.arange(len(img_subsections))[type_one_mask]

            #Get indices of all entries in type_one_mask (and therefore the indices of all subsections which are to be classified by our non-type-one classifier) 
            #   which are false. These are the subsections as input for the non-type-one classifier
            non_type_one_subsection_indices = np.arange(len(img_subsections))[np.logical_not(type_one_mask)]

            #Loop through subsection input indices for both models in batches, get the associated inputs in batches,
            #   then classify the batches (much faster than individual classification) and store back into our prediction grid.
            for type_one_subsection_i in range(0, len(type_one_subsection_indices), self.mb_n):
                sys.stdout.write("\rGenerating Prediction Grid on Image %i/%i. %.2f%% Complete." % (img_i, len(self.dataset.imgs)-1, 
                    (type_one_subsection_i/len(img_subsections))*100.0))

                #Get batch indices
                type_one_subsection_batch_indices = type_one_subsection_indices[type_one_subsection_i:type_one_subsection_i+self.mb_n]

                #Get batch inputs from the batch indices
                type_one_subsections = img_subsections[type_one_subsection_batch_indices]

                #Get batch outputs from the type one classifier
                type_one_predictions = self.type_one_classifier.predict(type_one_subsections)

                #Convert the local output classification enumeration of this classifier to the global ones and insert
                prediction_grid[type_one_subsection_batch_indices, 0:2] = type_one_predictions[:, 0:2]
                prediction_grid[type_one_subsection_batch_indices, 3] = type_one_predictions[:, 2]
                prediction_grid[type_one_subsection_batch_indices, 5] = type_one_predictions[:, 3]

            for non_type_one_subsection_i in range(0, len(non_type_one_subsection_indices), self.mb_n):
                sys.stdout.write("\rGenerating Prediction Grid on Image %i/%i. %.2f%% Complete." % (img_i, len(self.dataset.imgs)-1, 
                    ((non_type_one_subsection_i+len(type_one_subsection_indices)-1)/len(img_subsections))*100.0))

                #Get batch indices
                non_type_one_subsection_batch_indices = non_type_one_subsection_indices[non_type_one_subsection_i:non_type_one_subsection_i+self.mb_n]

                #Get batch inputs from the batch indices
                non_type_one_subsections = img_subsections[non_type_one_subsection_batch_indices]

                #Get batch outputs from the type one classifier
                non_type_one_predictions = self.non_type_one_classifier.predict(non_type_one_subsections)

                #Convert the local output classification enumeration of this classifier to the global ones and insert
                prediction_grid[non_type_one_subsection_batch_indices, 0] = non_type_one_predictions[:, 0]
                prediction_grid[non_type_one_subsection_batch_indices, 2:5] = non_type_one_predictions[:, 1:]


            #Reshape prediction grid to 2d now that we have all predictions, and denoise them.
            prediction_grid = np.reshape(prediction_grid, (prediction_h, prediction_w, self.class_n))
            prediction_grid = denoise_predictions(prediction_grid, self.denoising_weight)

            #Once denoised, save the predictions as argmaxed since we no longer need the full output
            self.before_editing[img_i] = np.argmax(prediction_grid, axis=2)
            self.after_editing[img_i] = np.argmax(prediction_grid, axis=2)

        sys.stdout.flush()
        print("")

        #Now that we've finished generating, we've started editing, so we update user progress.
        self.dataset.progress["prediction_grids_started_editing"] = True

    def edit(self):
        #Displays predictions on all images and allows the user to edit them until they are finished. The editor handles the saving of edits.
        editor = PredictionGridEditor(self.dataset)
