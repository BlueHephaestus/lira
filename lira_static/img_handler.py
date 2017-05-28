"""
Numerous functions for handling images and subsections in our LIRA programs,
    as well as a few for debugging said programs.

Further documentation found in each function.

-Blake Edwards / Dark Element
"""
import sys, os
import numpy as np
import cv2

def subsections_generator(img, sub_h, sub_w):
    """
    Arguments:
        img: a np array of shape (h, w, ...) 
        sub_h, sub_w: The size of each subsection that our image will be divided into when finished
    Returns:
        A generator, which yields a subsection each iteration, 
            by looping through the image and referencing each sub_h and sub_w subsection each time.
        We loop through 0 to (img.shape[0] // sub_h)*sub_h with a step of sub_h (and vice versa for sub_w)
        The important note here is the //, an integer division instead of a normal division. By doing this,
            we make sure we only loop through the subsections where row_i + sub_h is complete, not going
            outside the borders of our image.
        The key difference is that (img.shape[0]//sub_h)*sub_h is different from img.shape[0] only because
            the former no longer has any remainder when dividing it by sub_h, whereas img.shape[0] probably does.
        This is important because when there is a remainder, we would return that partial subsection from this generator,
            and that's not something we want to do because partials end up confusing our classifier.
        Of course, this is the same reasoning for sub_w.
    """
    for row_i in xrange(0, (img.shape[0]//sub_h)*sub_h, sub_h):
        for col_i in xrange(0, (img.shape[1]//sub_w)*sub_w, sub_w):
            yield img[row_i:row_i+sub_h, col_i:col_i+sub_w]

def get_subsections(sub_h, sub_w, img, rgb=False):
    """
    Arguments:
        sub_h, sub_w: The size of each subsection that our image will be divided into when finished
        img: a np array of shape (h, w, ...) where h % sub_h == 0 and w % sub_w == 0
        rgb: Boolean for if we are handling rgb images (True), or grayscale images (False).

    Returns:
        Loops through the img by row and column according to the sizes of our subsection,
            and places (sub_h, sub_w) size subsections into the resulting subs array.
        Returns a subs array of shape (h//sub_h, w//sub_w, sub_h, sub_w), 
            a matrix where each entry is the subsection at that location in the image.
    """
    """
    Initialize our subs array to size (h//sub_h, w//sub_w, sub_h, sub_w) for storing our subsections
    """
    if rgb:
        subs = np.zeros(shape=(img.shape[0]//sub_h, img.shape[1]//sub_w, sub_h, sub_w, 3))
    else:
        subs = np.zeros(shape=(img.shape[0]//sub_h, img.shape[1]//sub_w, sub_h, sub_w))

    """
    Use python's step in the range function to loop through our rows and columns by subsection height and width
    """
    for row in range(0, img.shape[0], sub_h):
        for col in range(0, img.shape[1], sub_w):
            """
            Get the subsection specified by our loops
            """
            sub = img[row:row+sub_h, col:col+sub_w]

            """
            Get the subsection-relative numbers. We don't need to bother with edge handling, because we already do so before this gets called.
            """
            row_i = row/sub_h
            col_i = col/sub_w

            """
            Place our subsection at our new row_i and col_i in our subs array
            """
            subs[row_i][col_i] = sub

    return subs

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
    Add our overlay to the img
    """
    return cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

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

def convert_2d_rects_to_1d_rects(rects_2d, width):
    """
    Arguments:
        rects_2d: 
            A np array of the format
                [x1_1, y1_1, x2_1, y2_1]
                [x1_2, y1_2, x2_2, y2_2]
                [ ...                  ]
                [x1_n, y1_n, x2_n, y2_n]

            Where each entry is the pairs coordinates corresponding to the top-left and bottom-right corners of a (2d) rectangle.
            These should be ints.
        width: Integer width of our image these rectangles belong to.

    Returns:
        rects_1d:
            A np array of the format

                [x1_1, x2_1]
                [x1_2, x2_2]
                [ ........ ]
                [x1_m, x2_m]

            Where each entry is the pairs coordinates corresponding to the left and right boundaries of new 1d rectangles, meaning 
                x1_1 will be the index of the first element in a certain rectangle, 
                and x2_1 will be the index of the last element.

            So that we can do array[x1_1:x2_1] for each pair, and only get elements that were originally in the rectangles.

            Due to the want for our array[x1_1:x2_1] condition, and the way the conversion works (detailed below), 
                we will have our rects_1d array be far longer than our original rects_2d array.
    """
    """
    Let's say we have a rectangle on an image. 
    This rectangle will be in the form of two points, like (x1, y1), (x2, y2).
    This rectangle has certain elements of our matrix (i.e. pixels of the image) within it.
    If we flatten our image, we'd like a way to reference these same elements with a sort of 1d rectangle.
    However, this isn't possible. There is no general way to get a 1d coordinate pair for this (of form (x1, x2)),
        given two 2d coordinates like (x1, y1), (x2, y2).

    Let's say we have an image with a rectangle in the top-left corner, so that the elements which are in the rectangle
        are marked by x's, and the ones not are marked by 0s.

        xxxxx000
        xxxxx000
        xxxxx000
        xxxxx000
        00000000
        00000000

    If we flatten the image, we end up with these elements:

        xxxxx000xxxxx000xxxxx000xxxxx0000000000000000000

    So we can see there is no one line of elements we can reference in the flattened image to get all the ones
        which were in the original image, in a a general case. 

    We need multiple 1d coordinates to do it. That's what this function does. Given some 2d coordinates,
        it goes through and gets all the 1d coordinates we need to fulfill this desired attribute.

    It does it by using the patterns inherent in this conversion,
        such as knowing our intervals will have (x2-x1)+1 elements,
        and knowing that there will be (y2-y1)+1 of these intervals.

        In our above example, we would have had:
            x1 = 0
            y1 = 0
            x2 = 4
            y2 = 3
            
            (4-0)+1 = 5 elements per interval
            (3-0)+1 = 4 intervals

    """ 
    """
    Our result list, since it's easier to just append
        rather than figure out how long the np array would be via
        all of our (y2-y1)+1 numbers, and then keep track of our
        position in this array, and so on.
    """
    rects_1d = []

    """ 
    Since it's easier to do this one rectangle at a time, that's how we do it.
    """
    for rect_2d in rects_2d:
        x1 = rect_2d[0]
        y1 = rect_2d[1]
        x2 = rect_2d[2]
        y2 = rect_2d[3]

        """
        Repeat ((y2-y1)+1) times
        """
        for i in range((y2-y1)+1):
            """
            Get 1d x1 normal way
            """
            x1_1d = (y1+i) * width + x1

            """
            Get 1d x2 using x1_1d + interval width,
                i.e. x1_1d + ((x2-x1)+1)
            """
            x2_1d = x1_1d + ((x2-x1)+1)
            
            """
            Then append these to new rects_1d list
            """
            rects_1d.append([x1_1d, x2_1d])

    """
    Then convert to an np array and cast to int,
        then return.
    """
    rects_1d = np.array(rects_1d).astype(int)

    return rects_1d
        
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
