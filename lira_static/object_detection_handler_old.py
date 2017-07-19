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
            detection_model: String file for our model, trained using HOG for object detection 
            model_dir: Directory of where all our models are stored. Will be used with `model_1`, `model_2`, and `detection_model` to obtain where the model is now located 

        Returns:
            Initializes our self.hog for getting HOG Descriptors on windows,
                and our self.model for classifying the HOG Descriptors as positive (object present) or negative (object not present) 
                when we do our sliding-window approach to generate bounding rectangles later.
        """
        model_dir = "%s%s%s.h5" % (model_dir, os.sep, detection_model)

        """
        Initialize hog descriptor using same parameters we used in get_hog_archive.py
        """
        win_shape = (512, 512)
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
        self.hog = cv2.HOGDescriptor(win_shape, block_size, block_stride, cell_size, nbins, deriv_aperture, win_sigma, histogram_norm_type, l2_hys_threshold, gamma_correction, n_levels)

        """
        Then load our model, which was trained for HOG descriptor input.
            We use the same technique as static_config.py, loading the model with Keras's method.
        """
        self.detector_model = load_model(model_dir)

    @staticmethod
    def pyramid(img, scale=.95, min_shape=[512,512], n=16):
        """
        Credit to here for the idea and basis for this function: http://www.pyimagesearch.com/2015/03/16/image-pyramids-with-python-and-opencv/
        It's a really good idea.

        Arguments:
            img: Our initial img, to be iteratively resized for each iteration of this generator.
            scale: The scale to resize our image up or down. 
                This is the value our image will be resized with, 
                    a value of 1.5 would increase it's size (i.e. 150% of original size),
                    and a value of 0.5 would decrease it's size (i.e. 50% of original size).
                It will be resized each iteration.
            min_shape: The img dimension sizes (a list of format [height, width]) at which we want to stop scaling our image down further, so as to avoid having an image too small for our windows.
                If we reach an image shape < min_shape, we will exit regardless of if we have reached n iterations.
                I did not include a max shape because it will never be too big for our windows, it would just make it take longer to scan windows across it.
            n: The number of times to scale our image down, can also be thought of as levels on the pyramid.

        Returns:
            Is a generator, initially yields the original (0, img) tuple, and each iteration afterwards resizes it using the scale factor, yielding that and the iteration number instead.
            Stops looping when either image dimension is smaller than it's associated min_shape counterpart or n iterations occur. Whichever occurs first.

            The tuple it returns is always of format (i, img), where i is the iteration in the loop. Will return 0 for the initial image, and then 1, 2, 3 and so on for the remaining loops.
        """
        #Just in case they give this as a tuple
        min_shape = list(min_shape)

        """
        Yield initial image
        """
        yield (0, img)

        """
        We loop over n - 1 so our generator yields exactly n times. Since we yield the initial image, this means we yield exactly n-1 resized versions.
        This of course assumes we don't exit prematurely due to a resized image smaller than our min_shape.
        """
        for i in range(n-1):
            img = cv2.resize(img, (0,0), fx=scale, fy=scale)
            if img.shape[0] < min_shape[0] or img.shape[1] < min_shape[1]:
                break
            yield (i+1, img)

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
    def non_max_suppression_fast(bounding_rects, overlap_threshold):
        """
        The following code is slightly modified from the code for this
            shown on this link: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

        The reason for this is because this is really good code, and was written there for the explicit
            purpose of doing fast non-maximum suppression, on bounding rectangles of the exact same format as
            the ones used in this file. So, if I didn't slightly modify it from there,
            I would have ended up with something either slower, or the same anyways.

        Arguments:
            bounding_rects: 
                A list of the format
                    [x1_1, y1_1, x2_1, y2_1]
                    [x1_2, y1_2, x2_2, y2_2]
                    [ ...                  ]
                    [x1_n, y1_n, x2_n, y2_n]

                Where each entry is the pairs coordinates corresponding to the top-left and bottom-right corners of a bounding rectangle.
                These should be floats.

            overlap_threshold:
                Float value to control how often we remove / suppress bounding rectangles based on overlap. 
                The smaller, the more we suppress, and vice versa.
        """
        """
        If there are no bounding_rects, return an empty list
        """
        if len(bounding_rects) == 0:
            return []

        """
        Otherwise we will need this as a np array
        """
        bounding_rects = np.array(bounding_rects)

        """
        Initialize the list of picked indexes	
        """
        pick = []

        """
        Grab the coordinates of the bounding_rects
        """
        x1 = bounding_rects[:,0]
        y1 = bounding_rects[:,1]
        x2 = bounding_rects[:,2]
        y2 = bounding_rects[:,3]

        """
        Compute the area of the bounding rects and sort the bounding
            rects by the bottom-right y-coordinate of each bounding rect
        """
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        """
        Keep looping while some indexes still remain in the indexes list
            i.e. until idxs is empty
        """
        while len(idxs) > 0:
            """
            Grab the last index in the indexes list and add the
                index value to the list of picked indexes
            """
            last = len(idxs) - 1
            i = idxs[-1]
            pick.append(i)

            """
            Find the largest (x, y) coordinates for the start (top-left) of
                the bounding rect and the smallest (x, y) coordinates
                for the end (bottom-right) of the bounding rect
            """
            xx1 = np.maximum(x1[i], x1[idxs[:-1]])
            yy1 = np.maximum(y1[i], y1[idxs[:-1]])
            xx2 = np.minimum(x2[i], x2[idxs[:-1]])
            yy2 = np.minimum(y2[i], y2[idxs[:-1]])

            """
            Compute the width and height of the bounding rect
            """
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            """
            Compute the ratio of overlap
            """
            overlap = (w * h) / area[idxs[:-1]]

            """
            Delete all indexes from the index list that overlap over our threshold
                np.delete deletes based on indices, which means it works well with np.where,
                    which returns the indices where a given condition for the elements of an array are true.
                Unfortunately np.delete doesn't yet support negative indexes,
                    so we have to do len(idxs) - 1 here instead of -1, which is what we used in the previous code.
            """
            idxs = np.delete(idxs, np.concatenate(([len(idxs)-1], np.where(overlap > overlap_threshold)[0])))

        """
        Return only the bounding rects that were picked
        """
        return bounding_rects[pick]

    def generate_bounding_rects(self, img):
        """
        Arguments:
            img: an array of shape (h, w, ...) where h % sub_h == 0 and w % sub_w == 0, our original main image

        Returns:
            Loops through different scales of our image, and slides a window across each of these scales,
                (making sure to have the same window shape as our HOG descriptor)
                and using each window as input to our HOG to get our HOG Descriptors. 

            Once we have our HOG descriptors, we use those as input to our model loaded from model_dir/detection_model.

            If we predict a positive example (i.e. object detected) with our model, we store it in our list of bounding rectangles,
                with our new entry being of format [x1, y1, x2, y1] 

            We then use non-maximum suppression on these bounding rectangles and return the new, suppressed bounding rectangles.

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
        Until further changes, we resize our images down by a constant resize factor.
        """
        #resize_factor = 0.05
        #resize_factor = 0.10
        #resize_factor = 0.2
        resize_factor = 0.25
        #resize_factor = 0.50
        img = cv2.resize(img, (0,0), fx=resize_factor, fy=resize_factor)

        """
        We use win_shape for both min_shape in pyramid(), and win_shape in sliding_window().
        This way, we don't need to check if our resized_img is larger than (or equal to) our window shape.
        """
        win_shape = [512, 512]
        scale = 0.9
        suppression_overlap_threshold = 0.1
        start = time.time()#For doing speed checks
        for (scale_i, resized_img) in self.pyramid(img, scale=scale, min_shape=win_shape, n=16):
            for (row_i, col_i, window) in self.sliding_window(resized_img, step_size=128, win_shape=win_shape):
                """
                row_i, col_i give us the top-left cord of our bounding rectangle on the resized image.
                The top-left cord is the same on the resized as the original image, 
                    however we will need to do some fancy computations using our scale_i in order to get the 
                    bottom-right cord of our bounding rectangle on the original image.
                Fortunately, we only need to care about coordinates if we predict a positive and need to store them,
                    so we don't need to do this on every window.
                """
                """
                We then get the HOG Descriptors of our window, 
                    Reshape them to be compatible with our detector model (i.e. same shape as input),
                    and get our prediction on these descriptors using our detector model.

                If the descriptors obtained on the window do not match the detector model's input size, this won't work.
                Be careful.
                """

                hog_descriptors = np.reshape(self.hog.compute(window), (1, -1))
                prediction = np.argmax(self.detector_model.predict_on_batch(hog_descriptors))
                if prediction:
                    """
                    We have a positive prediction.
                    This means we need to get the new window size for this prediction 
                        in order to properly get our lower-right corner coordinates on this bounding rectangle.
                    The window size is not changing on each of the iteratively resized images, 
                        but it is changing wrt the size of the original image size.
                    So if we keep making our images smaller, the window size wrt the original image will continue increasing in size.

                    In fact, I was able prove that it follows this rule (in paper):

                        New window size = original window size * (1 / scale_factor)**(iteration number)

                    For each dimension of the new (and original) window size.
                    Since we have the original window size, the scale factor, and the iteration number (scale_i), 
                        We compute the relative_win_size this way. 

                    We also do it on both dimensions at once by putting win_size into a np array.
                    """
                    print self.detector_model.predict_on_batch(hog_descriptors)
                    relative_win_shape = np.array(win_shape) * (1./scale)**(scale_i)

                    """
                    Using this, we offset our top-left coordinates and store the now complete set of coordinates
                        for our bounding rectangle into bounding_rects
                    """
                    #if np.max(self.detector_model.predict_on_batch(hog_descriptors)) >= .999:
                    bounding_rects.append([col_i, row_i, col_i+relative_win_shape[1], row_i+relative_win_shape[0]])


        print len(bounding_rects)
        """
        We then remove overlapping bounding rectangles using a non-maxima suppression algorithm
            (link for more info in the function)
        """
        #bounding_rects = self.non_max_suppression_fast(bounding_rects, suppression_overlap_threshold)
        print len(bounding_rects)

        """
        temp
        bounding_rects = np.array([[0,0,1500,1150]])
        #bounding_rects = np.array([[700, 700, 1000, 1000]])
        #bounding_rects = np.array([[0,0,300,300]])
        for (x1,y1,x2,y2) in bounding_rects:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite("test.png", img)
        end temp
        """

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
        """
        bounding_rects = np.array(bounding_rects)
        bounding_rects = bounding_rects * (1./resize_factor)

        """
        We then also cast it's elements to be of type int, 
            which we didn't do earlier because our calculations for the lower-right coordinate
            of each bounding rect often result in floating point values, 
            and you can't initialize python lists with a preset data type.
        We also get more float values when resizing it up.
        """
        bounding_rects = np.array(bounding_rects).astype(int)

        #print "%f seconds for detection" % (time.time() - start)

        return bounding_rects

"""
a = ObjectDetector("type1_detection_model", "../lira/lira2/saved_networks")
"""

"""
Small and quick test image
"""
"""
test = np.floor(np.random.rand(2048,2048,3)*255).astype(np.uint8)
#test = cv2.imread(os.path.expanduser("~/downloads/images/akihabara_background_1.jpg"))
a.generate_bounding_rects(test)
"""
"""
test = np.floor(np.random.rand(512,512,3)*255).astype(np.uint8)
a.generate_bounding_rects(test)
"""

"""
Full test image(s)
"""
#test1 = cv2.imread("../lira/lira1/data/rim_test_slides/115939_0_cropped.png")
#test1 = cv2.resize(test1, (1024, 1024))
"""
def test_on_img(i, f):
    test1 = cv2.imread(f)
    test1 = cv2.resize(test1, (0,0), fx=0.05, fy=.05)
    test1 = test1.astype(np.uint8)
    print test1.shape
    img_detected_bounding_rects = a.generate_bounding_rects(test1)
    print "%i Positives Detected" % len(img_detected_bounding_rects)
    print img_detected_bounding_rects
    for (x1,y1,x2,y2) in img_detected_bounding_rects:
        cv2.rectangle(test1, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite("%i.png" % i, test1)

for path in os.walk("../lira/lira1/data/rim_test_slides"):
    dir, b, fnames = path
    for i, fname in enumerate(fnames):
        fpath = dir + os.sep + fname
        test_on_img(i,fpath)
"""
