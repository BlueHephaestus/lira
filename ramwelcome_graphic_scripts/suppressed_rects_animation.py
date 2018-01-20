import cv2
import pickle
import numpy as np

def disp_img_fullscreen(img, name="test"):
    """
    Displays the given image full screen. 
    Usually used for debugging, uses opencv's display methods.
    """
    cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)          
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, 1)
    cv2.imshow(name,img)
    cv2.waitKey(1)

def list_find(l, e):
    for i,element in enumerate(l):
        if np.all(e == element):
            return i
    else:
        return -1
"""
Show all existing rects
Then can we outline each cluster as we check it?
Not easily. We would have to 1) find the rects in the cluster, and 2) outline such that only the outside was outlined.
So we can get our suppressed and unsuppressed, compare them to find what gets removed.
We should color the inside of the rectangle, too. 
We can go through and color each rectangle a different color depending on which, earlier we had the blue be bad and the red be good.
    Kinda like blue is passive but red is fiery, so we know red stands out and is good / what we want.
So we start them all with blue outlines, then we go through and change each to solid blue or red rectangles.
"""
#first load imgs, suppressed, and unsuppressed rects.
r = 0.031/0.2
for img_i in range(5):
    print "Image %i..." % img_i
    img = cv2.imread("img_%i.png" % img_i)
    resized_img = cv2.resize(img, (0,0), fx=r, fy=r)
    with open("unsuppressed_rects_%i.pkl"%img_i, "r") as f:
        unsuppressed_rects = pickle.load(f)
        unsuppressed_rects = 0.2 * unsuppressed_rects
        unsuppressed_rects = unsuppressed_rects.astype(int)
    with open("suppressed_rects_%i.pkl"%img_i, "r") as f:
        suppressed_rects = pickle.load(f)
        suppressed_rects = 0.2 * suppressed_rects
        suppressed_rects = suppressed_rects.astype(int)


    i = 0#Global i to use for our image filename
    #Blue outlines
    """
    print "Drawing Unsuppressed Rects..."
    for rect in unsuppressed_rects:
        resized_rect = (rect * r).astype(int)
        x1,y1,x2,y2 = resized_rect
        cv2.rectangle(resized_img, (x1, y1), (x2,y2), (0,0,255), 2)
        cv2.imwrite("images_for_gifs/%i/%07i.jpg"%(img_i,i), resized_img)
        i+=1
    """
    #usually should be a time.sleep here or else we'd just skip the previous step

    #Red solid rects for those from our previous animation
    print "Drawing Suppressed Rects..."
    for rect in unsuppressed_rects:
        resized_rect = (rect * r).astype(int)
        x1,y1,x2,y2 = resized_rect
        cv2.rectangle(resized_img, (x1, y1), (x2,y2), (0,0,255), -1)
        if i % 2 == 0:
            cv2.imwrite("images_for_gifs/%i/%07i.jpg"%(img_i,i), resized_img)
            pass
        i+=1

    #Blue solid rects for those in our unsuppressed_rects list, that we don't have in our suppressed_rects list (removed)
    #We first get all rects which are gonna get removed -> terminal_rects, by getting all that are in unsuppressed_rects list and not in suppressed_rects.
    terminal_rects = []
    for rect in unsuppressed_rects:
        if list_find(suppressed_rects,rect) < 0:
            terminal_rects.append(rect)

    #And now we draw these as blue solid rects
    print "Drawing Terminal Rects..."
    for rect in terminal_rects:
        resized_rect = (rect * r).astype(int)
        x1,y1,x2,y2 = resized_rect
        cv2.rectangle(resized_img, (x1, y1), (x2,y2), (255,0,0), -1)
        cv2.imwrite("images_for_gifs/%i/%07i.jpg"%(img_i,i), resized_img)
        i+=1

    for j in range(80):
        cv2.imwrite("images_for_gifs/%i/%07i.jpg"%(img_i,i), resized_img)
        i+=1




#done bitches
