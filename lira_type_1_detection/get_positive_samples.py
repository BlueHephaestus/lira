import os
import cv2 
import numpy as np

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
    #Just in case they give this as a tuple
    win_shape = list(win_shape)

    for row_i in xrange(0, img.shape[0], step_size):
        for col_i in xrange(0, img.shape[1], step_size):
            """
            Check to make sure we don't get windows which go outside of our image
            """
            if row_i + win_shape[0] <= img.shape[0] and col_i + win_shape[1] <= img.shape[1]:
                yield (row_i, col_i, img[row_i:row_i + win_shape[0], col_i:col_i + win_shape[1]])

shapes = []
i = 0
for path in os.walk("./raw_data"):
    dir, b, fnames = path
    for fname in fnames:
        fpath = dir + os.sep + fname
        img = cv2.imread(fpath)

        positives_dir = "./data/positives/"
        win_shape = [128, 128]
        scale = 0.7
        for (scale_i, resized_img) in pyramid(img, scale=scale, min_shape=win_shape, n=1):
            for (row_i, col_i, window) in sliding_window(resized_img, step_size=64, win_shape=win_shape):
                cv2.imwrite("%s%i.png" % (positives_dir, i), window)
                i+=1
                print i

