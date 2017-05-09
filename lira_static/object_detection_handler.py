import cv2
import numpy as np
import xml.etree.ElementTree as ET
import re
import numpy as np

class ObjectDetector(object):
    
    def __init__(self, svm_detection_model, model_dir):
        """
        Arguments:
            svm_detection_model: String .xml file for an OpenCV SVM, trained using HOG for object detection 
            model_dir: Directory of where all our models are stored. Will be used with `model_1`, `model_2`, and `svm_detection_model` to obtain where the model is now located 

        Returns:
            Initializes our self.svm with parameters located at model_dir/svm_detection_model.
        """
        model_dir = "%s%s%s.xml" % (model_dir, os.sep, svm_detection_model)

        """
        OpenCV 3.0 Made me do this. Seriously, I hate that I had to do this.
        The following code is copied from http://answers.opencv.org/question/56655/how-to-use-a-custom-svm-with-hogdescriptor-in-python/
        Pretty much word for word. Here's why I copied this, something which I hate doing and only do it in the case of a last resort:

        OpenCV 3.0 sucks.
            1) It's open source, but this doesn't mean it's well-commented. 
                Very often a method / function will have one or two short comments, and that will be it.
                Here's some examples: https://github.com/opencv/opencv/blob/master/samples/cpp/train_HOG.cpp

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

        """
        Initialize hog descriptor
        """
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(svmvec)#haha we LITERALLY CAN'T DO THIS BECAUSE WE CAN'T CHANGE THE WINDOW SIZE TO ANYTHING BUT THE DEFAULT HAH


    def generate_bounding_rectangles(img, svm_detection_model, model_dir):
        """
        Arguments:
            img: a np array of shape (h, w, ...) where h % sub_h == 0 and w % sub_w == 0, our original main image

        Returns:
            Uses OpenCV's detectMultiScale (parameters defined in this file) to slide a window across our img, using it as input to our 
                svm, loaded from model_dir/svm_detection_model.
            detectMultiScale also scales the image up (zooms in) until there is not enough room to slide the window or the maximum
                number of levels is reached.
            If the svm produces any positive examples, these bounding rectangles are then mean-shifted to remove overlapping positive examples / rectangles,
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


                
