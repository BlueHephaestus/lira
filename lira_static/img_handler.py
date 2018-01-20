"""
Numerous functions for handling images and subsections in our LIRA programs,
    as well as a few for debugging said programs.

Further documentation found in each function.

-Blake Edwards / Dark Element
"""
import sys, os
import numpy as np
import cv2

def get_subsections(sub_h, sub_w, img, rgb=False):
    """
    Arguments:
        sub_h, sub_w: The size of each subsection to be obtianed from our img argument.
        img: a np array of shape (h, w, 3) or (h, w) depending on the value of rgb
        rgb: Boolean for if we are handling rgb images (True), or grayscale images (False).

    Returns:
        Loops through the img by row and column according to the sizes of our subsection,
            and returns a (sub_h, sub_w) size subsection each iteration.
        This function is a generator, and returns the subsequent subsection each iteration,
            in the order left-right, top-bottom.
    """
    """
    Use python's step in the range function to loop through our rows and columns by subsection height and width.
    We do img.shape[0]-sub_h because we will be obtaining the sub via row:row+sub_h and col:col_sub_w
    """
    for row in range(0, img.shape[0]-sub_h, sub_h):
        for col in range(0, img.shape[1]-sub_w, sub_w):
            """
            Get the subsection specified by our loops
            """
            sub = img[row:row+sub_h, col:col+sub_w]

            """
            Yield this subsection
            """
            yield sub

def get_next_subsection(row_i, col_i, img_h, img_w, sub_h, sub_w, img, img_divide_factor):
    """
    Arguments:
        row_i, col_i: The row and column coordinates to get our big subsection from.
        img_h, img_w: The h and w of our img argument.
        sub_h, sub_w: The size of our individual subsections.
        img: a np array of shape (h, w, ...) where h % sub_h == 0 and w % sub_w == 0, our original main image
        img_divide_factor: The factor to divide our image by, to be used when determining the size of our return subsection.

    Returns:
        We use this to get big subsections from our (usually) massive img argument. 
            e.g. if img is size (24000, 40000), and img_divide_factor = 8, we would return a subsection of size (3000, 5000)
        This is so that we don't have to hold massive data structures in memory to accomodate for our usually massive img argument,
            and can instead hold something that is 1/img_divide_factor**2 the size.
        So, we determine the h and w of our new sub_img using this method, 
        then we get the absolute coordinates of where this sub_img is in our main img, 
        Make sure we get full subsections along the edges, 
        and then reference the full subsection and return it.
    """
    """
    Calculate initial h and w of our new sub_img
    """
    sub_img_h = img_h//img_divide_factor
    sub_img_w = img_w//img_divide_factor

    """
    We get the absolute coordinates of where this sub_img is in our main img
    """
    row = row_i * sub_img_h
    col = col_i * sub_img_w

    """
    We check if it's != 0 because if it is already divisible this will pointlessly use up more space.
        Otherwise, we only add on our extra subsection if it's not an edge. 
        This is because if it's an edge, we don't have any data to add to it.

    Using this method, we can get the entire edge subsections, and classify them correctly when feeding them into our model.
    """
    sub_img_h = sub_img_h - sub_img_h % sub_h
    sub_img_w = sub_img_w - sub_img_w % sub_w

    """
    So we check to make sure we don't already have the entire edge, and then we add on padding if not.
    Then, we finally check to make sure this big subsection is not at the edge of the full image, 
        so that we actually can pad without referencing outside the bounds of our image.
    """
    if sub_img_h % sub_h != 0:
        if row_i < img_divide_factor-1:
            sub_img_h += sub_h
    if sub_img_w % sub_w != 0:
        if row_i < img_divide_factor-1:
            sub_img_w += sub_w

    """
    Finally, get our sub img with our coordinates now that they are ready, and return.
    """
    sub = img[row:row+sub_img_h, col:col+sub_img_w]
    return sub

def add_weighted_overlay(img, overlay, alpha, rgb=False):
    """
    Arguments:
        img: np array of shape (h, w)
        overlay: np array of shape (h, w, 3), a BGR colored overlay to put on top of our original image.
        alpha: transparency weight of our overlay, percentage b/w 0 and 1, with 0 being no overlay and 1 being only overlay.
        rgb: Boolean for if we are handling rgb images (True), or grayscale images (False).

    Returns:
        if gray images (rgb=False):
            A new image, created by converting our img from greyscale to BGR, so that it goes from shape (h, w) to (h, w, 1) to (h, w, 3),
                the same shape as our overlay argument.
        if rgb images (rgb=True)
            We continue as normal, since we don't need to do any conversions.
            I am fairly certain that our images i've tested this on are RGB instead of BGR, and opencv automatically 
                converts it to the right one. However, if I am incorrect and our data is just in the proper format already, you can convert your data
                using "img = img[...,::-1]" to swap the channels.
        The overlay is then added onto the new (h, w, 3) img argument, with the alpha passed into opencv's addWeighted function.
        The result of this operation is returned, a combined image created by adding the overlay argument, weighted by alpha.
    """
    img_h = img.shape[0]
    img_w = img.shape[1]

    if not rgb:
        """
        Use our handy existing cv2 method to convert our gray 2d image into a 3d bgr one
        Note: Since it is greyscale, it doesn't matter if we go to RGB OR BGR
        """
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    """
    Set our overlay to np.uint8 if it isn't already, to have matching dtypes
    """
    overlay = overlay.astype(np.uint8)

    """
    Add our overlay to the img and return the result
    """
    return cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)

def clear_dir(dir):
    """
    Arguments:
        dir: A string directory to clear of all files and sub-directories.

    Returns:
        Clears/Deletes all files and sub-directories in the given dir, using python's os methods
        Has no return value.
    """
    for fname in os.listdir(dir):
        fpath = os.path.join(dir, fname)
        try: 
            if os.path.isfile(fpath):
                os.unlink(fpath)
        except:
            pass

def get_relative_factor(img_h, factor, threshold=3000, default_factor=8):
    """
    Arguments:
        img_h: Height of our image, used to compute our factor.
        factor: Usually not used in execution, however supplied for possibilities of modifications in the future, where a dynamic factor is not always viable/desirable.
        threshold: The default max size you expect the img_h argument to be in the result divided image.
        default_factor: The default factor to use, will be returned at the end if this is smaller than our computed value. Defaults to 8

    Returns:
        This will use the equation x/r = threshold,
            and return an integer representing the new factor to use.

        This is to tackle the problem of having really small images (e.g. 512x369) along with really big images (30kx67k),  
            since if we have such a small image our methods tend to mess up or be completely unnecessary, we can often
            have the entire thing in memory and save it entirely without need of splitting into subsections or resizing before saving(i.e. set r = 1)

        Using this equation, we can dynamically change the factor for each image, depending on the dimensions.
            Examples with threshold = 3000:
                (512, 369), 
                    512/r = 3000
                    512 = 3000r
                    512/3000 = r
                    ~0.1666 = r
                    In this case we'd set r = 1, since we can't have < 1
                    We don't bother with the other dimension
                (30835, 65686)
                    30835/3000 = r
                    10.27 = r
                    So we'd set r = 8, our default max
                (16800, 33600)
                    16800/3000 = r
                    5.6 = r
                    (taking ceil)
                    6 = r
                    So in this case we'd return r = 6 as our factor.

        In summary, it uses the result of the equation to determine if it should choose a value less than the default factor.
            Using this, our really big images will still be above the range enough to not bother it, but our really small images will get the r=1 that they need.

        r is limited to the range [1, default_factor]

        We return this result r, the factor.
    """
    #Get our factor
    factor = img_h/float(threshold)

    #Get it as an integer
    factor = int(np.ceil(factor))

    #Get the smaller of the two
    factor = min(default_factor, factor)

    return factor

def disp_img_fullscreen(img, name="test"):
    """
    Displays the given image full screen. 
    Usually used for debugging, uses opencv's display methods.
    """
    cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)          
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, 1)
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
