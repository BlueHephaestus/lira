from base import *
from UserProgress import UserProgress
from Images import Images
from TypeOneDetections import TypeOneDetections
from PredictionGrids import PredictionGrids

class Dataset(object):
    #To store all our subdatasets throughout our pipeline and manage progress throughout pipeline

    def __init__(self, uid=None, restart=None):
        #Get uid if needed
        if uid != None:
            self.uid = uid
        else:
            self.uid = input("Input your Unique/User ID for this Dataset: ")

        #Initialize user progress to existing progress if it exists and default starting progress otherwise
        self.progress = UserProgress(self.uid)

        #Check whether to reload the imgs archive, and possibly restart our progress
        if restart != None:
            self.restart = restart
        else:
            self.restart = input("Would you like to reset your classification progress and restart from the beginning? (This will re-load all images) [Y\\N]: ").upper()=="Y"

        if self.restart:
            #User wants to restart, both imgs and progress
            self.imgs = Images(restart=True)
            self.progress.restart()
            self.type_one_detections = TypeOneDetections(self, self.uid, restart=True)
            self.prediction_grids = PredictionGrids(self, self.uid, restart=True)

        else:
            #User does not want to restart. Defaults to this if they didn't put in "Y"
            if self.progress.editing_started():
                #If they were already editing these images, resume progress
                self.imgs = Images(restart=False)
                self.type_one_detections = TypeOneDetections(self, self.uid, restart=False)
                self.prediction_grids = PredictionGrids(self, self.uid, restart=False)
            else:
                #If they weren't already editing these images (i.e. they haven't started editing), load the images.
                #No need to restart our progress since it's already the initial value.
                self.imgs = Images(restart=True)
                self.type_one_detections = TypeOneDetections(self, self.uid, restart=True)
                self.prediction_grids = PredictionGrids(self, self.uid, restart=True)

    def detect_type_ones(self):
        #Detect type ones, suppress them, and allow human-in-the-loop editing.
        #If our user progress indicates they have already done some or all of these steps, we will skip over the already-completed steps.

        #Only generate if user hasn't started editing (meaning they already had them generated before)
        if not self.progress["type_ones_started_editing"]:
            self.type_one_detections.generate()

        #Only edit if the user hasn't finished editing
        if not self.progress["type_ones_finished_editing"]:
            self.type_one_detections.edit()

    def predict_grids(self):
        #Detect all predictions and allow human-in-the-loop editing
        #If our user progress indicates they have already done some or all of these steps, we will skip over the already-completed steps.

        #Only generate if user hasn't started editing (meaning they already had them generated before)
        if not self.progress["prediction_grids_started_editing"]:
            self.prediction_grids.generate()

        #Only edit if the user hasn't finished editing
        if not self.progress["prediction_grids_finished_editing"]:
            self.prediction_grids.edit()

    def get_data_stats(self):
        #Currently unimplemented. TODO AFTER
        pass
