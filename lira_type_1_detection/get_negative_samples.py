"""
Loop through all the images specified in our list,
    Scan a sub_hxsub_w window across each image as many times as possible,
    And save the window as a new negative sample after resizing down to dst_hxdst_w

The idea here is to use images that do not have any positive samples, 
    so as to get a representative distribution of negative samples by 
    splitting up these images into a bunch of smaller ones.

This is good because it's basically how we'll be scanning across our images when we use the detector,
    and as such is representative of the negative samples we'll encounter.

This is a really small script, and I may only use it once due to the extremely problem-specific nature of it..

-Blake Edwards / Dark Element
"""

import os
import numpy as np
import cv2

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
"""
Directory to store our new negative samples as we get them and resize them
"""
negatives_dir = "data/negatives/"

"""
Our image directory to use with our img_fnames in order to get img_paths
"""
img_dir = "../lira/lira1/data/full_test_slides_set"

"""
Our image filenames to use for this
"""
img_fnames = [
                "115892.svs",
                "115982.svs",
                "115907.svs",
                "116025.svs",
                "116000.svs",
                "115993.svs",
                "116024.svs",
                "115921.svs",

                "115885.svs",
                "115886.svs",
                "115887.svs",
                "115889.svs",
                "115891.svs",
                "115894.svs",
                "115897.svs",
                "115899.svs",
                "115901.svs",
                "115908.svs",
                "115910.svs",
                "115913.svs",
                "115918.svs",
                "115923.svs",
                "115931.svs",
                "115942.svs"
            ]

"""
Create an img_paths list by combining our directory with our filenames for each filename
"""
img_paths = [img_dir + os.sep + img_fname for img_fname in img_fnames]

"""
Our window size and destination image size after resize
"""
sub_h = 640
sub_w = 640
dst_h = 128
dst_w = 128

"""
Loop through images,
    with i as a global incrementer
"""
i = 0
for img_path in img_paths:
    img = cv2.imread(img_path)

    """
    Scan across usng our sliding window, using different scales with our pyramid, 
        using a window size of 2048 so we can resize down by 4 to our destination 512x512 size.
        we then save using our global incrementer as the filename.

    We make sure our step_size is 1/2 our window size
    """
    win_shape = [sub_h, sub_w]
    scale = 0.7
    for (scale_i, resized_img) in pyramid(img, scale=scale, min_shape=win_shape, n=1):
        for (row_i, col_i, window) in sliding_window(resized_img, step_size=int(sub_h/2.), win_shape=win_shape):
            window = cv2.resize(window, (dst_h, dst_w))
            cv2.imwrite("%s%i.png" % (negatives_dir, i), window)
            i+=1
            print i

"""
Make sure we get negative samples in the same way we view them when testing - with pyramid and sliding window.
"""

