import cv2
import os
import numpy as np
import xml.etree.ElementTree as ET
import re

class ObjectDetector(object):
    
    def __init__(self, svm_detection_model, model_dir):
        """
        Arguments:
            svm_detection_model: String .xml file for an OpenCV SVM, trained using HOG for object detection 
            model_dir: Directory of where all our models are stored. Will be used with `model_1`, `model_2`, and `svm_detection_model` to obtain where the model is now located 

        Returns:
            Initializes our self.hog using svm for detection - located at model_dir/svm_detection_model.
            We have to use a hog, since we need the detectMultiScale method of it. So that's why we don't call it self.detector, or something
        """
        model_dir = "%s%s%s.xml" % (model_dir, os.sep, svm_detection_model)

        """
        OpenCV 3.0 Made me do this. Seriously, I hate that I had to do this.
        The following code is copied from http://answers.opencv.org/question/56655/how-to-use-a-custom-svm-with-hogdescriptor-in-python/
        Pretty much word for word. Here's why I copied this, something which I hate doing and only do it in the case of a last resort:

        OpenCV 3.0 sucks.
            1) It's open source, but this doesn't mean it's well-commented. 
                Very often a method / function will have one or two short comments, and that will be it.
                Here's some examples (in one file): https://github.com/opencv/opencv/blob/master/samples/cpp/train_HOG.cpp

            2) It's also poorly documented. For many of the functions for HOG's and SVM's in OpenCV, the documentation pages for the methods 
                are either out of date, only available for C++, nonexistent, or very very undescriptive. 

                Sure, they have their help(cv2thingy), which is nice, but every time i've used it it shows the function signature and THAT'S IT.
                An example: 
                
                help(cv2.SVM) returns:
                ```
                Help on built-in function SVM in module cv2:

                SVM(...)
                    SVM([trainData, responses[, varIdx[, sampleIdx[, params]]]]) -> <SVM object>
                ```
                What do those arguments do? What is an SVM object? You sure don't know from this!
                I encourage you to try this with other methods (like any of the SVM ones or HOG ones) for more examples.

        Anyways. You can initialize a HOG Detector in OpenCV, and a big part of that is setting the SVM of that HOG. However,
            if you want to train your own SVM for this (in the case you want to detect anything other than pedestrians),
            you need to first get over all the hurdles OpenCV 3.0 throws in your way.
        Then, they have a very special format for setting the SVM detector of a HOG, which their default pedestrian detector is already equipped for.
        However, there seems to be literally no way built in to OpenCV, or documented in OpenCV, for doing this with ANY OTHER DETECTOR. 
        This is ridiculous, and the only way I could find to convert one of their xml files to the proper format was on the copy paste link above.

        So yea, OpenCV sucks. I hope you understand why i'm annoyed now.
        """
        tree = ET.parse(model_dir)
        root = tree.getroot()
        SVs = root.getchildren()[0].getchildren()[-2].getchildren()[0] 
        rho = float( root.getchildren()[0].getchildren()[-1].getchildren()[0].getchildren()[1].text )
        svmvec = [float(x) for x in re.sub( '\s+', ' ', SVs.text ).strip().split(' ')]
        svmvec.append(-rho)
        svmvec = np.array(svmvec)

        """
        Initialize hog descriptor using same parameters we used in get_hog_archive.py
        """
        win_size = (512, 512)
        block_size = (16,16)
        block_stride = (8,8)
        cell_size = (8,8)
        nbins = 9
        deriv_aperture = 1
        win_sigma = 4.
        histogram_norm_type = 0
        l2_hys_threshold = 2.0000000000000001e-01
        gamma_correction = False
        n_levels = 64
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins, deriv_aperture, win_sigma, histogram_norm_type, l2_hys_threshold, gamma_correction, n_levels)
        hog.setSVMDetector(svmvec)

        """
        Set our class attribute to this hog
        """
        self.hog = hog


    def generate_bounding_rectangles(self, img):
        """
        Arguments:
            img: a np array of shape (h, w, ...) where h % sub_h == 0 and w % sub_w == 0, our original main image

        Returns:
            Uses OpenCV's detectMultiScale (parameters defined in this file) to slide a window across our img, using it as input to our 
                hog, using an svm loaded from model_dir/svm_detection_model.
            detectMultiScale also scales the image up (zooms in) until there is not enough room to slide the window or the maximum
                number of levels is reached.
            If the detector produces any positive examples, these bounding rectangles are then mean-shifted to remove overlapping positive examples / rectangles,
                and at the end we return a list of the format
                    [(x1_1, y1_1), (x2_1, y2_1)]
                    [(x1_2, y1_2), (x2_2, y2_2)]
                    [ ...                      ]
                    [(x1_n, y1_n), (x2_n, y2_n)]
                Where each entry is a pair of coordinates corresponding to the top-left and bottom-right corners of the rectangle,
                    and each of those corners is denoted by an x and y coordinate pair.

                So we end up with an array of shape (n, 2, 2) where n is the total number of rectangles.
                We return this.

        """
        test = self.hog.detectMultiScale(img, winStride=(64, 64), scale=0.0, useMeanshiftGrouping=True)
        rects,weights = test
        print rects
        print rects.shape
        print type(rects[0])
        for (x, y, w, h) in rects:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        img = cv2.resize(img, (512,512))
        cv2.imshow("Detections", img)
        cv2.waitKey(0)

a = ObjectDetector("type1_detection_svm", "../lira/lira2/saved_networks")

test = np.floor(np.random.rand(2048,2048,3)*255).astype(np.uint8)
a.generate_bounding_rectangles(test)
"""
test = np.floor(np.random.rand(512,512,3)*255).astype(np.uint8)
a.generate_bounding_rectangles(test)
"""

"""
Load some test images for testing
"""
"""
test1 = cv2.imread("../lira/lira1/data/rim_test_slides/115939_0_cropped.png")
test1 = cv2.resize(test1, (0,0), fx=.25, fy=.25)
test1 = test1.astype(np.uint8)
print test1.shape
a.generate_bounding_rectangles(test1)
"""
