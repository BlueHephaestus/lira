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

def edit_detected_objects(rect_h=640, 
         rect_w=640, 
         step_h=320,
         step_w=320,
         dual_monitor=False,
         resize_factor=0.1,
         img_archive_dir="../lira/lira1/data/images.h5",
         rects_archive_dir="../lira/lira1/data/bounding_rects.pkl"
         ):
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
    Load the rects that our object detector detected for editing
    """
    with open(rects_archive_dir, "r") as f:
        rects = pickle.load(f)

    with h5py.File(img_archive_dir, "r", chunks=True, compression="gzip") as img_hf:
        """
        Get image number for iteration
        """
        img_n = len(img_hf.keys())

        """
        We want to save time by not re-initializing a big block of memory for our image
            every time we get a new image, so we find the maximum shape an image will be here.
        We do this by the product since blocks of memory are not in image format, their size 
            only matters in terms of raw number.
        We do the max shape because we initialize the img block of memory, and then initialize a bunch of other
            blocks of memory around this in the loop. 
        This means if we had less than the max, and then encountered a larger img than our current allocated block of memory,
            it would have to reallocate. 

        As my friend explains: "[Let's say] you initialize to the first image. Then you allocate more memory for other things inside the loop. That likely "surrounds" the first image with other stuff. Then you need a bigger image. Python can't realloc the image in place, so it has to allocate a new version, copy all the data, then release the old one."
        So we instead initialize it to the maximum size an image will be.
        """
        max_shape = img_hf.get("0").shape
        for img_i in range(img_n):
            img_shape = img_hf.get(str(img_i)).shape
            if np.prod(img_shape) > np.prod(max_shape):
                max_shape = img_shape

        """
        Now initialize our img block of memory to to this max_shape, 
            with the same data type as our images
        """
        img = np.zeros(max_shape, img_hf.get("0").dtype)

        """
        Start looping through images
        """
        for img_i in range(img_n):
            #Print progress
            print "Image %i/%i" % (i, img_n-1)

            """
            Get our image by resizing our block of memory instead of deleting and recreating it,
                and make sure to read directly from the dataset into this block of memory,
                so that we can reference it quickly when iterating through subsections for classification later.
            """
            img_dataset = img_hf.get(str(img_i))
            img.resize(img_dataset.shape)
            img_dataset.read_direct(img)

            """
            Now that we have our image, resize it with our resize factor
            """
            img = cv2.resize(img, (0,0), fx=resize_factor, fy=resize_factor)

            """
            Resize our rects list via elementwise rects * resize_factor, 
                then make sure it's an int because we need to draw using these values (e.g. we can't have 1.5 pixels),
                and convert it to a list so we can append and delete from it easily in our interactive_gui_handler.py
            """
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
                We also do the same process as earlier - except we have to convert to a np array, and also resize with 1/resize_factor instead of just resize_factor 
                    -to un-resize these rects so that they match the original resolution of the rects.
                Note: we also leave it as a nparray this time because we're done with these rects.
            """
            rects[i] = interactive_session.rects
            rects[i] = np.array(rects[i])
            rects[i] = rects[i] * (1/(resize_factor))
            rects[i] = rects[i].astype(int)

            """
            Set our flag back to false now that we are about to go to the next image.
            """
            interactive_session.flag_next=False

        """
        Now that we are finished editing the rects for every image,
            we save all our rects to overwrite the original rects_archive_dir file.
        """
        with open(rects_archive_dir, "w") as f:
            pickle.dump(rects, f)





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
#edit_detected_objects(rect_h=640, rect_w=640, step_h=320, step_w=320, resize_factor=0.1, img_archive_dir="../lira/lira1/data/images.h5", rects_archive_dir="../lira/lira1/data/bounding_rects.pkl", dual_monitor=True)
