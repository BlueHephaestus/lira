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

def main(rect_h=128, 
         rect_w=128, 
         dual_monitor=False,
         resize_factor=0.1,
         img_archive_dir="../lira/lira1/data/images.h5",
         rect_archive_dir="",
         dst_img_archive_dir="",
         dst_rect_archive_dir=""):
    """
    Open a new interactive session before starting our loops,
        and get the main canvas height and width from our session.
    We will use this for calculating our resize factor for each image later.
    """
    interactive_session = InteractiveGUI(rect_h, rect_w, dual_monitor)
    #canvas_h, canvas_w = interactive_session.main_canvas_height, interactive_session.main_canvas_width

    """
    Construct our entire testcases rects list to match the format it will be in our actual full implementation.
        They will start without being resized, since the resize factor will be different for every image - rect pair.
    """
    rects = []
    for i in range(8):
        with open("%i.pkl" % i, "r") as f:
            img_rects = pickle.load(f)
        img_rects = list(img_rects)
        rects.append(img_rects)

    """
    loop through and load each test case
    """
    for i in range(8):
        print i
        img = cv2.imread("testcase_%i.png" % i)

        """
        Our resize factor needs to be such that our final image height is the same as our canvas height, 
            so that we don't have vertical scroll and only have to worry about going horizontally.
        Since we want a resize factor r such that
            r * image height = canvas height, 
        We get r via
            r = (canvas height) / (image height)

        And we want to preserve the aspect ratio between our image height and width so we apply this ratio to both dimensions.
        """
        img_h = img.shape[0]
        #resize_factor = canvas_h / float(img_h)
        #img = cv2.resize(img, (0,0), fx=resize_factor, fy=resize_factor)#Execute resize

        #TEMP
        rects[i].append([0,0,256,256])
        #TEMP

        """
        Set our interactive UI's img and rects before starting the session and our main ui loop
        """
        #interactive_session.np_img = np.floor(np.random.rand(2000,2000,3)*255).astype(np.uint8)
        #interactive_session.np_img = (np.ones((2000,2000,3))*255).astype(np.uint8)
        interactive_session.np_img = img
        interactive_session.rects = rects[i]
        interactive_session.start_interactive_session()
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
        """
        rects[i] = interactive_session.rects

        """
        Set our flag back to false now that we are about to go to the next image.
        """
        interactive_session.flag_next=False

    """
    Now that we are finished with every image + rects, 
        we save all our updated data to the specified destination directories.
    """
    """
    Since I don't yet have this figured out, we are done. woot, woot.
    """







#img = cv2.resize(img, (0,0), fx=resize_factor, fy=resize_factor)
"""
for (x1,y1,x2,y2) in rects:
    cv2.rectangle(img, (x1,y1),(x2,y2), (0,0,255), 2)
#At this point image is ready for disp
"""
#disp_img_fullscreen(img)
#now display rects on img with rectangle method

main(rect_h=128, rect_w=128, img_archive_dir="../lira/lira1/data/images.h5", dual_monitor=True)
