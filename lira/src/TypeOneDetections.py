import sys, time
import cv2, h5py
import numpy as np
from keras.models import load_model

from base import *
from EditingDataset import EditingDataset
from TypeOneDetectionEditor import TypeOneDetectionEditor

class TypeOneDetections(object):
    def __init__(self, dataset, uid, restart=False):
        self.dataset = dataset#for reference, do not modify
        self.imgs = self.dataset.imgs
        self.uid = uid
        self.restart = restart

        #Our two attributes for detections before and after editing.
        self.archive_dir_before_editing = "../data/type_one_detections_before_editing/"#Where we'll store the .npy files for our detections before editing
        self.archive_dir_after_editing = "../data/type_one_detections_after_editing/"#Where we'll store the .npy files for our detections after editing
        self.before_editing = EditingDataset(self.dataset, self.uid, self.archive_dir_before_editing, restart=self.restart)
        self.after_editing = EditingDataset(self.dataset, self.uid, self.archive_dir_after_editing, restart=self.restart)

        #Detection parameters and classifier
        self.detection = True

        #Size of each side of the cross
        self.side_size = 2000

        #Stride to move the cross each iteration
        self.cross_stride = 500

        #Size of each lengthwise segment 
        self.segment_size = 200

        #Stride of our segment blocks for each check
        self.segment_stride = 100

        #Scalar to weight our standard deviations by before applying operations to them.
        self.stdw = 0.5

        #Threshold to ensure we don't scan segments across empty slide
        self.empty_slide_threshold = 210

        #Threshold for how high the mean of a detection is allowed to be
        self.intensity_threshold = 140

        #Kernel for our Gaussian Blur
        self.gaussian_kernel = (7,7)

    def generate(self):
        #Generates detections for each image, saving each detection array to self.before_editing.

        #Quick lambda for use later
        dist = lambda a,b: abs(a-b)

        #Generate for each image
        for img_i, img in enumerate(self.imgs):
            
            #Progress indicator
            sys.stdout.write("\rGenerating Type One Detections on Image {}/{}...".format(img_i, len(self.imgs)-1))

            #final detections for img, will be archived after detection and suppression
            detections = []

            if self.detection:
                #Get only the green channel and apply our gaussian blur to it
                img = cv2.GaussianBlur(img[:,:,1],self.gaussian_kernel,0)

                h,w = img.shape[0],img.shape[1]

                #Iterate through crosses (not doing multiple scales yet)
                for y in range(self.side_size,h-self.side_size,self.cross_stride):
                    for x in range(self.side_size,w-self.side_size,self.cross_stride):
                        bot =   img[y:y+self.side_size, x] 
                        top =   img[max(0,y-self.side_size):y, x][::-1]#Reverse this so we iterate through it from the origin outwards
                        right = img[y, x:x+self.side_size]
                        left =  img[y, max(0,x-self.side_size):x][::-1]#Reverse this so we iterate through it from the origin outwards

                        cross = [top,right,bot,left]
                        detection = [False,False,False,False]#Will be rectangle coordinates, x1, y1, x2, y2

                        #Iterate through the 4 sides as independent vectors
                        for i, side in enumerate(cross):
                            #Slide our segment window across the side with our size and stride
                            n = len(range(0, len(side)-2*self.segment_size, self.segment_stride))
                            for segment_step in range(0, len(side)-2*self.segment_size, self.segment_stride):
                                #Get mean and std of all 3 segments
                                segments = [
                                            side[segment_step+0*self.segment_size:segment_step+1*self.segment_size], 
                                            side[segment_step+1*self.segment_size:segment_step+2*self.segment_size], 
                                            side[segment_step+2*self.segment_size:segment_step+3*self.segment_size]
                                           ]
                                means = [np.mean(segment) for segment in segments]
                                stds = [np.std(segment)*self.stdw for segment in segments]

                                #Ensure we don't have empty slide stuff
                                if max(means[0],means[2]) < self.empty_slide_threshold:
                                    #If both means are within the smallest std of the two from each other
                                    if dist(means[0],means[2]) < max(stds[0],stds[2]):
                                        #And the middle mean's range is below their combined std ranges
                                        if means[1] < min(means[0]-stds[0], means[2]-stds[2]):
                                            #We have a detection here, mark it for this side and end our loop here
                                            detection[i] = segment_step+3*self.segment_size
                                            break

                        #If we have at least 3 that detect a rim
                        c = 0
                        for side in detection:
                            if side:
                                c+=1
                        if c >= 3:
                            #If we have exactly 3 that detect a rim but not 4
                            if c == 3:
                                #Set the 4th to be the mirror of it's adjacent side
                                if not detection[0]:
                                    detection[0] = detection[2]
                                elif not detection[1]:
                                    detection[1] = detection[3]
                                elif not detection[2]:
                                    detection[2] = detection[0]
                                elif not detection[3]:
                                    detection[3] = detection[1]
                            
                            #Create detection rectangle coordinates from our cross coordinates
                            x1 = x-detection[3]
                            y1 = y-detection[0]
                            x2 = x+detection[1]
                            y2 = y+detection[2]
                            #If this detection's mean is below the intensity threshold
                            if np.mean(img[y1:y2,x1:x2]) < self.intensity_threshold:
                                detection = [x1,y1,x2,y2]
                                detections.append(detection)

            #Save these detections to both the before and after editing datasets, since we initialize them to be the same.
            self.before_editing[img_i] = detections
            self.after_editing[img_i] = detections

        #Now that we've finished generating, we've started editing, so we update user progress.
        self.dataset.progress["type_ones_started_editing"] = True

        sys.stdout.flush()
        print("")

    def edit(self):
        #Displays detections on all images and allows the user to edit them until they are finished. The editor handles the saving of edits.
        editor = TypeOneDetectionEditor(self.dataset)
        #editor.start_editing()


