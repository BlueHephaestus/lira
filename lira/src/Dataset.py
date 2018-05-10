import sys

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

    def get_stats(self):
        #Once we're sure the user's session is complete:
        if self.progress["prediction_grids_finished_editing"]:

            #Generate a CSV, with several statistics detailed below. The data for the CSV is first created in a (len(self.imgs)+1, 8) shape numpy array.
            #This is because we have one extra column to store the number of type one clusters per image, 
            #   one extra column for the row titles,
            #   minus one column since we don't include empty slide,
            #   and one extra row for the average across all images for all columns.
            stats = np.zeros((len(self.imgs)+1, 8), dtype="U20")

            #Additional array to keep track of classification counts (not including empty slide) so we can compute average percentages later
            classification_counts = np.zeros((6))

            #To keep track of detection counts to compute normal averages
            detection_counts = 0

            #Loop through each image predictions and detections, inserting stats computed as we go along 
            for i, (prediction_grid, detections) in enumerate(zip(self.prediction_grids.after_editing, self.type_one_detections.after_editing)):
                sys.stdout.write("\rGenerating Stats on Image {}/{}...".format(i, len(self.imgs)-1))

                #Set Image number
                stats[i][0] = i+1

                #Increment counts and get percentages of each classification on this image
                col_i = 0
                valid_prediction_n = np.sum(prediction_grid!=3)
                for classification in range(7):
                    #Don't include empty slide in any of our calculations
                    if classification!=3:
                        count = np.sum(prediction_grid==classification)
                        classification_counts[col_i]+=count
                        stats[i][col_i+1] = "%.8f%%" % (count/valid_prediction_n*100)
                        col_i+=1

                #Get and Insert the number of Type One Lesions / Type One Detection Clusters in this image
                detection_count = len(get_rect_clusters(detections))
                detection_counts+=detection_count
                stats[i][-1] = detection_count

            #For the final row, use our classification_counts array to insert the average percentage of each classification type across all images
            stats[-1][0] = "Average"
            col_i = 0
            for classification in range(7):
                if classification != 3:
                    stats[-1][col_i+1] = "%.8f%%" % (classification_counts[col_i]/np.sum(classification_counts)*100)
                    col_i+=1

            #Also insert the average number of Type One Lesions / Type One Detection Clusters in this image using our counter
            stats[-1][-1] = detection_counts/len(self.imgs)

            #Save it to a CSV with the appropriate header / column names in the "Output Stats" directory
            np.savetxt("../../Output Stats/{}_stats.csv".format(self.uid), stats, fmt="%s,%s,%s,%s,%s,%s,%s,%s",
                    header="Image,Healthy Tissue,Type I - Caseum,Type II,Type III,Type I - Rim,Unknown/Misc,Number of Type One Lesions", comments="")

            sys.stdout.flush()
            print("")


