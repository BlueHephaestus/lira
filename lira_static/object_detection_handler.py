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
        """
        Just in case they give this as a tuple, we cast to list
        """
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

        Note: As this function makes use of recursion, there are several helper functions and lambdas inside this function to simplify things.
        """
        def check_rect_connected_to_rect(check_rect, rect):
            """
            Arguments:
                check_rect: Rect we are checking, to see if it is connected to rect.
                    Should be of format [x1, y1, x2, y2]
                rect: Rect we are checking our check_rect against.  
                    Should be of format [x1, y1, x2, y2]
                Note: check_rect and rect may be of different sizes, this function still works in this case.
            
            Returns:
                True if our check_rect is bordering or overlapping / connected to our rect,
                False if not.

                Does a few simple if statements on the coordinates of our variables to check all possible cases
                    where our rectangles may be bordering or connected, and if none of those return True, we return False
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

        """
        We will use this for recording which rectangles have been checked already.
        If a rectangle does not exist in the dictionary, it hasn't been checked.
        If a rectangle does exist with value False, then it has been checked but so far isn't part of a cluster of size >= cluster_threshold.
        If a rectangle does exist with value True, then it has been checked and is part of a cluster of size >= cluster_threshold.
        We will only set values to True in the last part of the function, once we've finished creating our connected_rects list.
        """
        checked_rects = {}

        """
        This will be our new list of rectangles. Given a list, we remove all rectangles in clusters of size < cluster_threshold,
            and the remaining rectangles are added to this list as the .
        """
        suppressed_bounding_rects = []

        """
        Quick function to get a string for a given rectangle,
            since the rectangles are lists and as such can't be used as dictionary keys.
        """
        rect_to_key = lambda rect: ','.join(map(str, rect))

        def get_connected_rects(rect, rects):
            """
            Arguments:
                rect: Our given rectangle, we will check all rectangles in rects to see if this rectangle is connected with any others.
                rects List of rectangles to search through for connected rectangles of our rect argument.

            Returns:
                suppressed_bounding_rects: List of all rectangles which are connected to our given rect rectangle. 
                    They may also be bordering, but we're going to refer to them as connected.
                Given a rectangle, this function will find all connected rectangles of that rectangle, 
                    then call itself on all of those to get their connected rectangles, and so on until all connected rectangles (or nodes if you want to think of it that way)
                    have been found. 
                So, this function will return all rectangles our given rect is connected to, if those rectangles have not been checked yet.
                We have to make sure they haven't been checked yet, otherwise we may continuously recurse onto rectangles we've already got in our list.
                Using this function, we can get the entire cluster of interconnected rectangles (aka a "subgraph" if you want to think of it that way) in a list,
                    for us to easily check the size of the cluster as well as the rectangles inside the cluster.
            """
            connected_rects = []

            """
            Loop through our rectangles, and see if any of them are connected to this rectangle and also haven't been checked yet.
                If they are, add them to our connected_rects. 
            Note: We also avoid checking the caller's rectangle by doing this.
            """
            for check_rect in rects:
                check_rect_key = rect_to_key(check_rect)
                if (check_rect_connected_to_rect(check_rect, rect) and (check_rect_key not in checked_rects.keys())):
                    checked_rects[check_rect_key] = False
                    connected_rects.append(check_rect)

            """
            Now that we have all the rectangles our rect is directly connected to, we want to get 
                all the rectangles those rectangles are directly connected to, and repeat until
                all connected rectangles have been obtained.
            So, we call this function on all directly connected rectangles, extending our original list to include
                all rectangles returned from these calls (adding nothing if [] is returned),
                and then return our list of connected_rects to ensure we recurse correctly to our caller function.
            Remember: Our list of connected_rects is full of rects which have not yet been checked yet.
                If we had rects which had been checked already in this list, we could recurse indefinitely. So we wanna avoid that.
            """
            for connected_rect in connected_rects:
                connected_rects.extend(get_connected_rects(connected_rect, rects))

            return connected_rects

        """
        For every rectangle in our given list of bounding rectangles (bounding_rects),
        """
        for i, rect in enumerate(bounding_rects):
            """
            Check if the rect has been checked yet, 
            """
            rect_key = rect_to_key(rect)
            if rect_key not in checked_rects.keys():
                """
                And if not, set it to checked and 
                    get all it's connected rects using our get_connected_rects function. 
                """
                checked_rects[rect_key] = False 
                connected_rects = get_connected_rects(rect, bounding_rects)

                """
                Also add our original rect to this connected_rects list as well, since 
                    we don't wanna forget the source rect.
                """
                connected_rects.append(rect)

                """
                Now that we have a list of all connected rects including the original rect,
                    we have a list for our cluster.
                We then check if the length (i.e. size) of our cluster is >= cluster_threshold,
                    and if so we know that these rectangles are rectangles we want to keep.
                """
                if len(connected_rects) >= cluster_threshold:
                    """
                    So, we set all of their values in the checked_rects dictionary to True
                        because we want to keep them
                    """
                    for connected_rect in connected_rects:
                        rect_key = rect_to_key(connected_rect)
                        checked_rects[rect_key] = True

        """
        Now we have a dictionary where every rect in a cluster size >= cluster_threshold is labeled with True,
            and all other rectangles are labeled with False.
        So we loop through all our rects again and append all the ones we want to keep to our resulting suppressed_bounding_rects
            list, and return this list.
        """
        for rect in bounding_rects:
            rect_key = rect_to_key(rect)
            if checked_rects[rect_key]:
                suppressed_bounding_rects.append(rect)
        return suppressed_bounding_rects

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
        print "Before Suppression: %i" % len(bounding_rects)
        bounding_rects = self.suppress_by_cluster_size(bounding_rects, win_shape, 30)
        print "After  Suppression: %i" % len(bounding_rects)

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
