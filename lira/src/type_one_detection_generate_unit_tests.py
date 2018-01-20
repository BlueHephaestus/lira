import cv2, sys
import numpy as np
#Generates on some images without suppression, generates on some images with suppression
#Should be able to look in the unit tests directory and see the results to judge them yourself

#We pretty much just have the detection_suppression option primarily so that we can easily test with and without. But it also could be useful to others.
from TypeOneDetections import TypeOneDetections
from Dataset import Dataset
dataset = Dataset()
dataset.type_one_detections.detection_suppression = False
dataset.type_one_detections.detection_resize_factor = 0.2
dataset.detect_type_ones()
dataset.progress["type_ones_started_editing"] = False
#Loop through images and detections without suppression
i = 0
print("Before Suppression")
for img, detections in zip(dataset.imgs, dataset.type_one_detections.before_editing):
    img = np.array(img)
    #create img of these
    for detection in detections:
        cv2.rectangle(img, tuple(detection[0:2]), tuple(detection[2:4]), (0, 0, 255), 12)
    img = cv2.resize(img, (0,0), fx=.1, fy=.1)
    cv2.imwrite("../data/unit_tests/type_one_detections_before_suppression_{}.png".format(i), img)

    i+=1

#Loop through images and detections with suppression
dataset.type_one_detections.detection_suppression = True
dataset.detect_type_ones()
i = 0
print("After Suppression")
for img, detections in zip(dataset.imgs, dataset.type_one_detections.before_editing):
    #create img of these
    for detection in detections:
        cv2.rectangle(img, tuple(detection[0:2]), tuple(detection[2:4]), (0, 0, 255), 12)
    img = cv2.resize(img, (0,0), fx=.1, fy=.1)
    cv2.imwrite("../data/unit_tests/type_one_detections_after_suppression_{}.png".format(i), img)

    i+=1
