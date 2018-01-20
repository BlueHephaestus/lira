import sys
import cv2
import numpy as np
import pickle
import time
import imageio

def disp_img_fullscreen(img, name="test"):
    """
    Displays the given image full screen. 
    Usually used for debugging, uses opencv's display methods.
    """
    cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)          
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, 1)
    cv2.imshow(name,img)
    cv2.waitKey(1)

"""
Loop through with blue windows, check if each window is the right rect, if so leave it as a red marker.
First we load unsuppressed rects and normal imgs, then start our big loop
"""
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

def list_find(l, e):
    for i,element in enumerate(l):
        if np.all(e == element):
            return i
    else:
        return -1


r = 0.031/0.2
#win_shape = (r*np.array([128,128])).astype(int)
win_shape = np.array([128,128])
#step_size = int(r*128/2.)
step_size = 64
for img_i in range(5):
    img = cv2.imread("img_%i.png" % img_i)
    resized_img = cv2.resize(img, (0,0), fx=r, fy=r)
    with open("unsuppressed_rects_%i.pkl"%img_i, "r") as f:
        unsuppressed_rects = pickle.load(f)
        unsuppressed_rects = 0.2 * unsuppressed_rects
        unsuppressed_rects = unsuppressed_rects.astype(int)

    #For print progress
    window_n = float(img.shape[0]//step_size * img.shape[1]//step_size)

    i = 0 
    #with imageio.get_writer('unsuppressed_rects.gif', mode='I') as writer:
    for (row_i, col_i, window) in sliding_window(img, step_size=step_size, win_shape=win_shape):
        #Print % progress
        sys.stdout.write("\rImage {} -> {:.2%} Complete".format(img_i, i/window_n))
        sys.stdout.flush()

        resized_img_clone = resized_img.copy()
        rect = np.array([col_i, row_i, col_i+win_shape[1], row_i+win_shape[0]])
        resized_rect = (rect * r).astype(int)

        #Blue rectangle - because opencv uses BGR instead of RGB FOR SOME REASON
        cv2.rectangle(resized_img_clone, (resized_rect[0], resized_rect[1]), (resized_rect[2], resized_rect[3]), (255,0,0), 2)
        if list_find(unsuppressed_rects,rect) >= 0:
            #Red rectangle - because opencv uses BGR instead of RGB FOR SOME REASON
            cv2.rectangle(resized_img, (resized_rect[0], resized_rect[1]), (resized_rect[2], resized_rect[3]), (0,0,255), 2)
            cv2.rectangle(resized_img_clone, (resized_rect[0], resized_rect[1]), (resized_rect[2], resized_rect[3]), (0,0,255), 2)

        #Write every other one because it gets sped up so fast that the loss of quality is not noticeable.
        if i % 20 == 0:
            #writer.append_data(resized_img_clone)
            cv2.imwrite("images_for_gifs/%i/%07i.jpg"%(img_i,i), resized_img_clone)

        i+=1
