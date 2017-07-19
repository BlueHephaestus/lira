import cv2
import numpy as np
import os

import keras
from keras.models import load_model

import time

class ObjectDetector(object):
    
    def __init__(self, detection_model, model_dir):
        """
        Arguments:
            detection_model: String file for our model, trained for object detection 
            model_dir: Directory of where all our models are stored. Will be used with `model_1`, `model_2`, and `detection_model` to obtain where the model is now located 

        Returns:
            Initializes our self.detector_model for classifying input samples as positive (object present) or negative (object not present) 
                when we do our sliding-window approach to generate bounding rectangles later.
        """
        model_dir = "%s%s%s.h5" % (model_dir, os.sep, detection_model)

        """
        Then load our model, which was trained for our input.
            We use the same technique as static_config.py, loading the model with Keras's method.
        """
        self.detector_model = load_model(model_dir)


    @staticmethod
    def sliding_window(img, step_size=64, win_shape=[512,512]):
        """
        Credit to here for the idea and basis for this function: http://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
        It's a really good idea. And that guy is also awesome and deserves free snacks.

        Arguments:
            img: Our initial img, to have a window iteratively scanned across for each iteration of this generator.
            step_size: The step size, amount we move our window each iteration.
            win_shape: The exact amount of which you want to win. Just kidding.
                This is a [height, width] formatted list for the shape of our window which we are scanning across the img argument.

        Returns:
            Is a generator, yields a tuple of format (row_i, col_i, window)  each iteration.
            This window will be of shape win_shape, and it will step across the img in both the x and y directions (w and h dimensions)
                with step_size step size.
        """
        #Just in case they give this as a tuple, we cast to list
        win_shape = list(win_shape)

        for row_i in xrange(0, img.shape[0], step_size):
            for col_i in xrange(0, img.shape[1], step_size):
                """
                Check to make sure we don't get windows which go outside of our image
                """
                if row_i + win_shape[0] <= img.shape[0] and col_i + win_shape[1] <= img.shape[1]:
                    yield (row_i, col_i, img[row_i:row_i + win_shape[0], col_i:col_i + win_shape[1]])

    @staticmethod
    def suppress_by_cluster_size(bounding_rects, win_shape, cluster_threshold):
        """
        Arguments:
            bounding_rects: 
                A list of the format
                    [x1_1, y1_1, x2_1, y2_1]
                    [x1_2, y1_2, x2_2, y2_2]
                    [ ...                  ]
                    [x1_n, y1_n, x2_n, y2_n]

                Where each entry is the pairs coordinates corresponding to the top-left and bottom-right corners of a bounding rectangle.
                These should be floats.
            win_shape: Shape of our window.
            cluster_threshold: If the size of a cluster (determined by # of rects in the cluster) is lower than this number, it gets removed. Otherwise, it stays.

        Returns:
            suppressed_bounding_rects:
                A list of the same format as bounding_rects, however should only have clusters of rectangles where the number of rects is >= cluster_threshold

            Yes, I made this myself. Yes, it's awesome.
        """
        checked_rects = {}
        suppressed_rects = []
        def check_rect_connected_to_rect(check_rect, rect):
            """
            Arguments:
                check_rect: Rect we are checking, to see if it is connected to rect.
                    Should be of format [x1, y1, x2, y2]
                rect: Rect we are checking our check_rect against.  
                    Should be of format [x1, y1, x2, y2]
                Note: check_rect and rect may be of different sizes, this function still works in this case.
            """
            """
            We then get the rect's coordinates as variables for easy reference
            """
            check_x1, check_y1, check_x2, check_y2 = check_rect
            x1, y1, x2, y2 = rect

            """
            In the next part, it helps to remember that:
            x2 = x1 + window width
            y2 = y1 + window height
            """

            """
            Check if top-left corner of check rect is inside our main rect
            """
            if ((check_x1 >= x1 and check_x1 <= x2) and (check_y1 >= y1 and check_y1 <= y2)):
                return True

            """
            Check if top-right corner of check rect is inside our main rect
            """
            if ((check_x2 >= x1 and check_x2 <= x2) and (check_y1 >= y1 and check_y1 <= y2)):
                return True

            """
            Check if bot-left corner of check rect is inside our main rect
            """
            if ((check_x1 >= x1 and check_x1 <= x2) and (check_y2 >= y1 and check_y2 <= y2)):
                return True

            """
            Check if bot-right corner of check rect is inside our main rect
            """
            if ((check_x2 >= x1 and check_x2 <= x2) and (check_y2 >= y1 and check_y2 <= y2)):
                return True

            """
            If none of the above were true, then it's not touching our main rect, and therefore not connected.
            """
            return False

        rect_within_window = lambda rect, cord, window: 0 <= np.abs(cord - check_cord) <= window 
        rect_to_key = lambda rect: ','.join(map(str, rect))
        def get_connected_rects(rect, rects):
            """
            rect is specific rectangle
            rects is list of all rectangles
            connected_rects is our list we are building
            """
            #print "Getting connected rects..."
            connected_rects = []
            """
            Loop through our rectangles, and see if any of them are connected to this rectangle and also haven't been checked yet.
                If they are, add them to our list. 
            Note: We also avoid checking the caller's rectangle by doing this.
            """
            for check_rect in rects:
                #Actual check
                check_rect_key = rect_to_key(check_rect)
                if (check_rect_connected_to_rect(check_rect, rect) and (check_rect_key not in checked_rects.keys())):
                    #if (cord_within_boundary(check_rect[0], rect[0], win_shape[1]) or cord_within_boundary(check_rect[2], rect[2], win_shape[1])) and (cord_within_boundary(check_rect[1], rect[1], win_shape[0]) or cord_within_boundary(check_rect[3], rect[3], win_shape[0])):
                    checked_rects[check_rect_key] = False
                    connected_rects.append(check_rect)
            for connected_rect in connected_rects:
                #Only call this function if the rect is not yet checked
                #connected_rect_key = rect_to_key(connected_rect)
                """
                print "\t",connected_rect
                print "\t",connected_rect_key
                print "\t",checked_rects
                print ""
                """
                #if connected_rect_key not in checked_rects.keys():
                    #checked_rects[connected_rect_key] = False
                tmp = get_connected_rects(connected_rect, rects)
                connected_rects.extend(tmp)
                    #print "\t",len(tmp),len(connected_rects)
                    #sys.exit()
                #sys.exit()

            #print "\tExiting",len(connected_rects)
            #print "Finished getting connected rects..."
            return connected_rects

        """
        import cv2
        img = np.zeros((6000, 13000, 3))
        """
        for i, rect in enumerate(bounding_rects):
            rect_key = rect_to_key(rect)
            if rect_key not in checked_rects.keys():
                checked_rects[rect_key] = False 
                """
                print "\t",rect_key
                print "\t",checked_rects
                print ""
                """
                connected_rects = get_connected_rects(rect, bounding_rects)
                #print "\t",connected_rects
                #print "\t",len(connected_rects)
                """
                Draw an image with all rects and original rect to check
                """
                """
                cv2.rectangle(img, (rect[0], rect[1]),(rect[2],rect[3]), (255, 0, 0), 3)
                for rect in connected_rects:
                    cv2.rectangle(img, (rect[0], rect[1]),(rect[2],rect[3]), (0, 0, 255), 3)
                """

                """
                if len(connected_rects)>0:
                    sys.exit()
                """
                """
                for connected_rect in connected_rects:
                    connected_rect_key = rect_to_key(connected_rect)
                    if connected_rect_key not in checked_rects.keys():
                        checked_rects[connected_rect_key] = False
                """
                connected_rects.append(rect)
                if len(connected_rects) >= cluster_threshold:
                    for connected_rect in connected_rects:
                        rect_key = rect_to_key(connected_rect)
                        checked_rects[rect_key] = True
        #print "**",len(checked_rects)
        #cv2.imwrite("debugging.jpg", img)
        #Now we have a dict where every one in a cluster of proper size is labeled with a 1
        for rect in bounding_rects:
            rect_key = rect_to_key(rect)
            if checked_rects[rect_key]:
                suppressed_rects.append(rect)
        return suppressed_rects

    def generate_bounding_rects(self, img):
        """
        Arguments:
            img: an array of shape (h, w, ...) where h % sub_h == 0 and w % sub_w == 0, our original main image

        Returns:
            Slides a window across our image, and uses each window as input to our model loaded from model_dir/detection_model.

            If we predict a positive example (i.e. object detected) with our model, we store it in our list of bounding rectangles,
                with our new entry being of format [x1, y1, x2, y1] 

            We then (optionally) use a suppression algorithm on these bounding rectangles and return the new, suppressed bounding rectangles.

            At the end we return a list of the format
                [x1_1, y1_1, x2_1, y2_1]
                [x1_2, y1_2, x2_2, y2_2]
                [ ...                  ]
                [x1_n, y1_n, x2_n, y2_n]

            Where each entry is the pairs coordinates corresponding to the top-left and bottom-right corners of the bounding rectangle,

            So we end up with an array of shape (n, 2*2) where n is the total number of rectangles.
            We return this.
        """
        """
        Initialize our list of bounding rectangles (aka bounding rectangle coordinates)
        """
        bounding_rects = []

        """
        Cast img to nparray just in case it isn't already
        """
        #img = np.array(img)

        """
        We resize our images down by a constant resize factor, 
            since these lesions are noticed by lowering the resolution / zooming out of the image, 
            and looking at it from a larger perspective. Because of this, we don't need the full resolution image,
            as it both doesn't make sense because we want to zoom out and get a bigger perspective,
            and also because it greatly increases the computational cost.
        
        This is similar to those games where you have to identify what an object is but the object is very zoomed in,
            and it slowly zooms out to give you more details. These lesions are a lot easier to classify when 
            we zoom out first, and since we're only looking for macroscopic details (since we're zoomed out), 
            we can afford to lower the resolution.
        """
        #resize_factor = 0.05
        #resize_factor = 0.10
        resize_factor = 0.2#1/5th
        #resize_factor = 0.25
        #resize_factor = 0.50
        img = cv2.resize(img, (0,0), fx=resize_factor, fy=resize_factor)

        win_shape = [128, 128]
        start = time.time()#For doing speed checks

        """
        Slide across windows of size win_shape, with step size of step_size
        """
        for (row_i, col_i, window) in self.sliding_window(img, step_size=64, win_shape=win_shape):
            """
            row_i, col_i give us the top-left cord of our bounding rectangle on the image.
            """
            prediction = np.argmax(self.detector_model.predict_on_batch(np.array([window])))
            if prediction:
                """
                We have a positive prediction.
                So we add our full set of coordinates as top-left, bottom-right by adding the window shape 
                    to our top-left coordinates.
                """
                bounding_rects.append([col_i, row_i, col_i+win_shape[1], row_i+win_shape[0]])


        print len(bounding_rects)
        """
        We then use our suppress_by_cluster_size suppression algorithm 
            to remove all clusters of rectangles with size less than our passed in cluster_threshold.
        Clusters are groups of interconnected/bordering rectangles, where the size of a cluster 
            is the number of interconnected / bordering rectangles in the group.
        Our cluster_threshold was arbitrary, and was found to be the optimal number for our samples, 
            since many of the false positives were in small clusters of rectangles of size < cluster_threshold,
            and so far all of the true positives have been in clusters of rectangles >= cluster_threshold.
            So, it worked really well to remove a lot of false positives and greatly improve our accuracy.
        """
        bounding_rects = self.suppress_by_cluster_size(bounding_rects, win_shape, 30)
        print len(bounding_rects)

        """
        Since this bounding_rects was obtained on an image which was resized down by resize_factor 
            (a number of format 1/x, so always 0 < 1/x <= 1, i.e. 1/20 = 0.05),
        We want to resize the coordinates to match the original img argument. 
        Fortunately, since the number is of format 1/x, a downscale (large image -> smaller image) resize,
            we just need the upscale (smaller image -> large image) resize number, which is just x.
        So how do we get x from 1/x? We just do 1/(1/x) = x, by simple algebra.
        And we can just resize the coordinates back up by multiplying them by this x.

        Since we need array multiplication for this (and will need it in future ops once this is returned),
            we cast it before applying the multiplication.

        Since this multiplication results in floats, we then cast the elements to be of type int afterwards.
        """
        bounding_rects = np.array(bounding_rects)
        bounding_rects = bounding_rects * (1./resize_factor)
        bounding_rects = np.array(bounding_rects).astype(int)

        #For speed checks
        #print "%f seconds for detection" % (time.time() - start)

        return bounding_rects
