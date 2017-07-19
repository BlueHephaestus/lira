import object_detection_handler
from object_detection_handler import *
import numpy as np
import cv2

a = ObjectDetector("phase2_smaller_augmented_samples_type1_detection_model", "../lira/lira2/saved_networks")

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
def show_rects(i, f):
    img = cv2.imread(f)
    resize_factor = 0.50
    img = cv2.resize(img, (0,0), fx=resize_factor, fy=resize_factor)
    print img.shape

    win_shape = [512, 512]
    scale = 0.7

    for (scale_i, resized_img) in pyramid(img, scale=scale, min_shape=win_shape, n=16):
        print resized_img.shape
        i = 0
        for (row_i, col_i, window) in sliding_window(resized_img, step_size=128, win_shape=win_shape):
            i+=1
            if i <= 100:

                clone = resized_img.copy()
                cv2.rectangle(clone, (col_i, row_i), (col_i + 512, row_i + 512), (0, 255, 0), 100)
                clone = cv2.resize(clone, (0,0), fx=0.05, fy=0.05)
                cv2.imshow("Window", clone)
                cv2.waitKey(1)

def test_on_img(i, f):
    test1 = cv2.imread(f)
    #test1 = cv2.resize(test1, (0,0), fx=0.05, fy=.05)
    #test1 = test1.astype(np.uint8)
    #print test1.shape
    img_detected_bounding_rects = a.generate_bounding_rects(test1)
    print "%i: %i Positives Detected" % (i, len(img_detected_bounding_rects))
    #print img_detected_bounding_rects
    for (x1,y1,x2,y2) in img_detected_bounding_rects:
        cv2.rectangle(test1, (x1, y1), (x2, y2), (0, 255, 0), 2)
    test1 = cv2.resize(test1, (0,0), fx=0.05, fy=.05)
    cv2.imwrite("%i.jpg" % i, test1)

#test_on_img(0, "../lira/lira1/data/test_slides/115890.svs")
for path in os.walk("../lira/lira1/data/full_test_slides_set"):
    dir, b, fnames = path
    for i, fname in enumerate(fnames):
        fpath = dir + os.sep + fname
        show_rects(i,fpath)
        sys.exit()
