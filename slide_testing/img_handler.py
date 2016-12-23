
import sys, os
import numpy as np
import cv2

def get_subsections(sub_h, sub_w, img, verbose):
    """
    Divide our given img into subsections of width sub_w and height sub_h
    """
  
    #Set it to resulting size
    subs = np.zeros(shape=(img.shape[0]//sub_h, img.shape[1]//sub_w, sub_h, sub_w))#Final array of subsections

    #Use stride length to get our subsections to splice on
    sub_i = 0
    sub_total = ((img.shape[0]//sub_h)+1)*((img.shape[1]//sub_w)+1)-1
    for row in range(0, img.shape[0], sub_h):
        for col in range(0, img.shape[1], sub_w):
            #Get the subsection specified by our loops
            
            sub = img[row:row+sub_h, col:col+sub_w]

            """
            Get the subsection-relative numbers. We don't need to bother with edge handling, because we already do so before this gets called.
            """
            row_i = row/sub_h
            col_i = col/sub_w

            subs[row_i][col_i] = sub

            if verbose:
                sys.stdout.write("\r\tSubsection #%i / %i" % (sub_i, sub_total))

            sub_i += 1
    if verbose:
        print ""#For flush print formatting
    return subs

def get_next_subsection(row_i, col_i, img_h, img_w, sub_h, sub_w, img, img_divide_factor):

    """
    Get the subsection specified by our loops.

    Our sub_img_h and sub_img_w are the resulting dimensions for the overrall new big subsection, 
        since we are using this to separate the massive images into more manageable subsection images.

    We calculate the remaining space and add it to our sub_img_h and sub_img_w so that we always get full subsections.
        Since we previously pad the image with zeros, we should never get problems, even on the far edges.
    """
    #Calculate initial sizes
    sub_img_h = img_h//img_divide_factor
    sub_img_w = img_w//img_divide_factor

    #Get cords
    row = row_i * sub_img_h
    col = col_i * sub_img_w

    """
    Now set it to a size that is %sub_h and %sub_w == 0
        We have to handle this differently if sub_img_h < sub_h or sub_img_w < sub_w,
            because e.g. 70 % 80 = 70
    Now we have to handle for several cases here.
        If either dimension is less than the subsection dimension, e.g. sub_img_h < sub_h,
            we just set it equal to the dimension of the subsection if it's not an edge.
            Otherwise we leave it be.
        If it's not,
            We check if it's != 0 because if it is already divisible this will pointlessly use up more space.
                Otherwise, we only add on our extra subsection if it's not an edge. 
                This is because if it's an edge, we don't have any data to add to it.
    """
    """
    if sub_img_h < sub_h:
        if row_i < img_divide_factor-1:
            sub_img_h = sub_h
    else:
        sub_img_h = sub_img_h - sub_img_h % sub_h
        if sub_img_h % sub_h != 0:
            if row_i < img_divide_factor-1:
                sub_img_h += sub_h

    if sub_img_w < sub_w:
        if row_i < img_divide_factor-1:
            sub_img_w = sub_w
    else:
        sub_img_w = sub_img_w - sub_img_w % sub_w
        if sub_img_w % sub_w != 0:
            if row_i < img_divide_factor-1:
                sub_img_w += sub_w
    """
    """
    We check if it's != 0 because if it is already divisible this will pointlessly use up more space.
        Otherwise, we only add on our extra subsection if it's not an edge. 
        This is because if it's an edge, we don't have any data to add to it.
    """
    sub_img_h = sub_img_h - sub_img_h % sub_h
    sub_img_w = sub_img_w - sub_img_w % sub_w

    if sub_img_h % sub_h != 0:
        if row_i < img_divide_factor-1:
            sub_img_h += sub_h
    if sub_img_w % sub_w != 0:
        if row_i < img_divide_factor-1:
            sub_img_w += sub_w

    sub = img[row:row+sub_img_h, col:col+sub_img_w]
    return sub

def pad_img(img_h, img_w, sub_h, sub_w, img):
    """
    Pads our image with enough zeros so that we never have the problem of partials
        on the far edges.
    """
    new_img_h = img_h - img_h % sub_h + sub_h
    new_img_w = img_w - img_w % sub_w + sub_w

    img = np.lib.pad(img, ((0, new_img_h-img_h), (0, new_img_w-img_w)), 'constant', constant_values = 0) 
    return img


def clear_dir(dir):
    for fname in os.listdir(dir):
        fpath = os.path.join(dir, fname)
        try: 
            if os.path.isfile(fpath):
                os.unlink(fpath)
        except:
            pass

def get_relative_factor(img_h, factor, threshold=3000, default_factor=8):
    """
    This will use the equation x/r = threshold,
        and return an integer representing the new factor to use.

    This is to tackle the problem of having really small images (e.g. 512x369) along with really big images (30kx67k),  
        since if we have such a small image our methods tend to mess up or be completely unnecessary, we can often
        have the entire thing in memory and save it entirely without need of splitting into subsections or resizing before saving(i.e. set r = 1)

    Using this equation, we can dynamically change the factor for each image, depending on the dimensions.
        Example with threshold = 3000:
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
        Using this, our big images will still be above the range enough to not bother it, but our small images will get the r=1 that they need.

    r is limited to the range [1, default_factor]
    """

    #Get our factor
    factor = img_h/float(threshold)

    #Get it as an integer
    factor = int(np.ceil(factor))

    #Get the smaller of the two
    factor = min(default_factor, factor)

    return factor

def disp_img_fullscreen(img):
    cv2.namedWindow("test", cv2.WND_PROP_FULLSCREEN)          
    cv2.setWindowProperty("test", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
    cv2.imshow("test",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
