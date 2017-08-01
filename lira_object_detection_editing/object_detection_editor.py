"""
"""
import pickle
import cv2
import numpy as np

from interactive_gui_handler import *

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

def main(rect_h=640, 
         rect_w=640, 
         step_h=320,
         step_w=320,
         dual_monitor=False,
         resize_factor=0.1,
         img_archive_dir="../lira/lira1/data/images.h5",
         rect_archive_dir="",
         dst_img_archive_dir="",
         dst_rect_archive_dir=""):
    """
    Make sure that rect_h and rect_w match the resolution on the full res images, not the input to the object detector.
        This means if you resize your images down to 1/5th their original size before feeding 128x128 windows 
        into an object detector, you should have rect_h = rect_w = 128 * 5, since when you resize the images back up to 
        their full resolution you are doing a resize_factor = 1/(1/5) = 5. So your 128x128 rects are now 128*5x128*5 = 640x640.
    Same goes for step_h and step_w
    Be careful!
    """
    """
    Open a new interactive session before starting our loops,
        and get the main canvas height and width from our session.
    We will use this for calculating our resize factor for each image later.
    """
    interactive_session = InteractiveGUI(dual_monitor)
    canvas_h, canvas_w = interactive_session.main_canvas_height, interactive_session.main_canvas_width

    """
    Construct our entire testcases rects list to match the format it will be in our actual full implementation.
        They will start without being resized, since the resize factor will be different for every image - rect pair.
    """
    rects = []
    for i in range(8):
        with open("new_%i.pkl" % i, "r") as f:
            img_rects = pickle.load(f)
        img_rects = list(img_rects)
        rects.append(img_rects)

    """
    loop through and load each test case
    """
    for i in range(8):
        #MAKE A PRINT STATEMENT FOR PROGRESS HERE IN THE FULL IMPLEMENTATION
        print "Image %i/%i" % (i, 
        print i
        img = cv2.imread("testcase_%i.png" % i)

        """
        Convert our rects list for this image into a numpy array so we have easy elementwise multiplication of a 2d list,
            then resize our rects list via elementwise rects * resize_factor, 
            then make sure it's an int because we need to draw using these values (e.g. we can't have 1.5 pixels),
            and convert it back to a list
        """
        rects[i] = np.array(rects[i])
        rects[i] = rects[i] * resize_factor
        rects[i] = rects[i].astype(int)
        rects[i] = list(rects[i])

        """
        Set our interactive UI's img, rects, rect_h, and rect_w before starting the session and our main ui loop
        """
        interactive_session.np_img = img
        interactive_session.rects = rects[i]

        #These need to be casted to an int because we need to draw using these values
        interactive_session.rect_h = int(rect_h * resize_factor)
        interactive_session.rect_w = int(rect_w * resize_factor)
        interactive_session.step_h = int(step_h * resize_factor)
        interactive_session.step_w = int(step_w * resize_factor)


        #Finally start the session
        interactive_session.start_interactive_session()

        #And finally start the main ui loop
        while True:
            """
            Main loop where we wait for flag(s) from Interactive UI before doing anything else
            """
            if interactive_session.flag_next:
                """
                If our flag is set to go to the next image in our interactive session, we break out of our wait/main/flag loop to go to the next image + rects list.
                """
                break

        """
        Get our final rects list for this image from our InteractiveGUI object, and update our main rects list with the new rects for this image.
            This is why we have our main rects list as a list, so we can just replace the existing list with a new list which may be of completely different size.
            We also do the same process as earlier - but with 1/resize_factor instead of just resize_factor - to un-resize these rects so that they match the original resolution of the rects.
        """
        rects[i] = interactive_session.rects
        rects[i] = np.array(rects[i])
        rects[i] = rects[i] * (1/(resize_factor))
        rects[i] = rects[i].astype(int)
        rects[i] = list(rects[i])

        """
        Set our flag back to false now that we are about to go to the next image.
        """
        interactive_session.flag_next=False

    """
    Now that we are finished with every image + rects, 
        we save all our updated data to the specified destination directories.
    """
    for i in range(8):
        with open("new_%i.pkl" % i, "w") as f:
            pickle.dump(rects[i], f)





"""
Pre-conditions for calling this function
    called after we have our images, AND RECTS.
    we know where our object_detection_handler is.
    We loop through images in our archive,
    loop through their associated rects,
    resize each image down by a constant amount resize_factor(in this case we'd set this to 0.1), 
    resize each rects down by the same factor,
    then go on with the rest of our code.
    Each iteration of the loop we get the next image and rects.
    then 

So our pre conditions are that we have an .h5 for our images,
    and a .pkl for our rects.
So we need a script to put our images through for getting the detected objects.
Probably should put this in our object_detection_handler, if we can.
K we handled it, needed a new script for it.
But anyways now we have both images and rects.
"""
main(rect_h=640, rect_w=640, step_h=320, step_w=320, resize_factor=0.1, img_archive_dir="../lira/lira1/data/images.h5", rects_archive_dir="../lira/lira1/data/bounding_rects.pkl", dual_monitor=True)
