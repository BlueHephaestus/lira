import sys, time
import cv2, h5py
import numpy as np
from keras.models import load_model

from base import *
from EditingDataset import EditingDataset

class TypeOneDetections(object):
    def __init__(self, dataset, uid):
        self.dataset = dataset#for reference, do not modify
        self.imgs = self.dataset.imgs
        self.uid = uid

        #Our two attributes for detections before and after editing.
        self.archive_dir_before_editing = "../data/type_one_detections_before_editing/"#Where we'll store the .npy files for our detections before editing
        self.archive_dir_after_editing = "../data/type_one_detections_after_editing/"#Where we'll store the .npy files for our detections after editing
        self.before_editing = EditingDataset(self.dataset, self.uid, self.archive_dir_before_editing)
        self.after_editing = EditingDataset(self.dataset, self.uid, self.archive_dir_after_editing)

        #Detection parameters and classifier
        self.detection_resize_factor = 0.2
        self.detection_suppression = True
        self.detection_step_size = 64
        self.detection_window_shape = (128, 128)#On the RESIZED image
        self.detection_cluster_threshold = 30#For suppression
        self.detection_classifier = load_model("../classifiers/type_one_detection_classifier.h5")#For detection

    def generate(self):
        #Generates detections and suppresses them for each image, saving each detection array to self.before_editing.

        #Generate for each image
        for i, img in enumerate(self.imgs):
            #Progress indicator
            sys.stdout.write("\rGenerating Type One Detections on Image {}/{}...".format(i, len(self.imgs)-1))

            #resize img down for detection
            img = cv2.resize(img, (0,0), fx=self.detection_resize_factor, fy=self.detection_resize_factor)

            #final detections for img, will be archived after detection and suppression
            detections = []

            #scan model input window across our now resized image
            for (row_i, col_i, window) in windows(img, self.detection_step_size, self.detection_window_shape):
                #If classifier predicts positive (we use nparray to add one dimension to make 4d instead of 3d)
                if np.argmax(self.detection_classifier.predict(np.array([window]))):
                    #Add window as detection in format [x1, y1, x2, y2]
                    detections.append([col_i, row_i, col_i+self.detection_window_shape[1], row_i+self.detection_window_shape[0] ])

            if self.detection_suppression:
                #suppress detections for this image based on rectangle cluster size
                detections = get_rect_clusters(detections)

                #Remove clusters < detection_cluster_threshold 
                detections = [cluster for cluster in detections if not len(cluster) < self.detection_cluster_threshold]

                #Reshape list of clusters of rects into list of rects nx4 
                detections = [rect[:] for cluster in detections for rect in cluster]

            #Convert to np array and resize detections to match original image, and cast to int.
            detections = (np.array(detections)/self.detection_resize_factor).astype(int)

            #Save these detections to the before editing dataset.
            self.before_editing[i] = detections

            #Now that we've finished generating, we've started editing, so we update user progress.
            self.dataset.progress["type_ones_started_editing"] = True

        sys.stdout.flush()
        print("")

    def edit(self):
        """
        displays, edits, saves

        """
        pass
