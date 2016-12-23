
import sys
import numpy as np
import cv2

def get_subsections(sub_h, sub_w, img, verbose):
    #Divide our given img into subsections of width sub_w and height sub_h
  
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
            If we have an edge section(where the dimensions are less because we get cut off), pad it with zeros the necessary amount
                Correction: This never happens now. They are always equal.
            if (sub.shape[0] < sub_h or sub.shape[1] < sub_w):
                sub = np.lib.pad(sub, ((0, sub_h-sub.shape[0]), (0, sub_w-sub.shape[1])), 'constant', constant_values = 0) 
            """
            #Get the subsection-relative numbers
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

    #Now set it to a size that is %sub_h and %sub_w == 0
    """
    if row_i < img_divide_factor-1:
        sub_img_h = sub_img_h - sub_img_h % sub_h 
    else:
        sub_img_h = sub_img_h - sub_img_h % sub_h

    if col_i < img_divide_factor-1:
        sub_img_w = sub_img_w - sub_img_w % sub_w
    else:
        sub_img_w = sub_img_w - sub_img_w % sub_w 
    """
    """
    if row_i == 0 and col_i == 0:
        sub_img_h = sub_img_h - sub_img_h % sub_h 
        sub_img_w = sub_img_w - sub_img_w % sub_w 
        sub = img[row:row+sub_img_h, col:col+sub_img_w]
    else:
        col = col - sub_img_w % sub_w
        print col, sub_img_w % sub_w

        sub_img_h = sub_img_h - sub_img_h % sub_h
        sub_img_w = sub_img_w - sub_img_w % sub_w 
        sub = img[row:row+sub_img_h, col:col+sub_img_w]
    """
    """
    if row_i > 0:
        row = row - sub_img_h % sub_h
    if col_i > 0:
        col = col - sub_img_w % sub_w
    """

    sub_img_h = sub_img_h - sub_img_h % sub_h
    sub_img_w = sub_img_w - sub_img_w % sub_w 

    #We check if it's != 0 because if it is already divisible this will pointlessly use up more space.
    if sub_img_h % sub_h != 0:
        if row_i < img_divide_factor-1:
            sub_img_h += sub_h

    if sub_img_w % sub_w != 0:
        if col_i < img_divide_factor-1:
            sub_img_w += sub_w

    sub = img[row:row+sub_img_h, col:col+sub_img_w]

    #cv2.imwrite('test/%i_%i.jpg' % (row_i, col_i), sub)
    #print sub.shape
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
    

def divide_img(img, factor):
    sub_h = img.shape[0]//factor
    sub_w = img.shape[1]//factor
    print img.shape
    print sub_h
    print sub_w
    return get_subsections(sub_w, sub_h, img)
