"""
The main function for this file is edit_detected_objects(), so the bulk of the documentation for this file lies there.

-Blake Edwards / Dark Element
"""
import pickle
import h5py
import cv2
import numpy as np

from interactive_gui_handler import *

def edit_detected_objects(rect_h=640, 
         rect_w=640, 
         step_h=320,
         step_w=320,
         dual_monitor=False,
         images_already_resized=False,
         resize_factor=0.1,
         img_archive_dir="../lira/lira1/data/images.h5",
         resized_img_archive_dir="../lira/lira1/data/resized_images.h5",
         rects_archive_dir="../lira/lira1/data/bounding_rects.pkl"
         ):
    """
    Arguments:
        rect_h and rect_w: The size of our individual detected rectangles in our image.
        step_h and step_w: The size of our step size or stride length between each rectangle in our image.
            (the minimal distance possible between two rects)
        dual_monitor: Boolean for if we are using two monitors or not. 
            Shrinks the width of our display if we are, and leaves normal if not.
        images_already_resized: Boolean for if we've already resized our images before. Defaults to False.
        resize_factor: The scale to resize our image by, e.g. 0.5 will make a 400x400 image into a 200x200 image.
            This should always be 0 < resize_factor <= 1.0
        img_archive_dir: a string filepath of the .h5 file where the images are stored.
        resized_img_archive_dir: a string filepath of the .h5 file where the resized images will be stored.
        rects_archive_dir: a string filepath of the .pkl file where the detected rects will be stored.

    Returns:
        Uses interactive_gui_handler.py (in the lira_object_detection_editing directory) to open a custom UI,
            to allow the user to edit any objects (in our case Type 1 lesions) our object detector detects.
            -> For our case, these "detections" are Type 1 lesions.

        Detections are in the form of rectangle coordinates, and our UI will display these rectangles to allow users to add/remove
            rectangles. By doing this, they modify the detections. So you'll see me refer to both of them, but they're the same thing.
 
        It uses a similar structure to LIRA Live, where we instantiate an InteractiveUI object, set it's image, rectangle, and other data attributes,
            then start the interactive session. 

        It runs this session, waiting for flags from interactive_gui_handler, until our interactive_gui_handler catches that our user wants to go to the next image, 
            at which point this function will exit out of this image, save the rectangles from this image, and go to the next one.
        
        It repeats this process until all images have been exhausted.

        We used to resize the images one at a time, as they were encountered in this big image loop. 
            However, I noted while beta testing that it took ~45s to load each image, 
                so instead we now resize them all beforehand and put them into resized_img_archive_dir. 
            This way, the user can actually do something (lunch, nap, graduate thesis) with the hour or so it takes beforehand, 
                instead of being stuck waiting for a minute on each image, which would be hell.

        After we're done, we save our new rects to rects_archive_dir, a .pkl file.

        Do not forcibly close or interrupt the UI, as any of your progress will be lost.
            Note: If this does happen, after the resizing is complete i.e. once the UI is open, you can comment that part out and run again to skip the resizing part - since that archive should be ready.

        NOTE:
            Make sure that rect_h and rect_w match the resolution on the full res images, not the input to the object detector.
                This means if you resize your images down to 1/5th their original size before feeding 128x128 windows 
                into an object detector, you should have rect_h = rect_w = 128 * 5, since when you resize the images back up to 
                their full resolution you are doing a resize_factor = 1/(1/5) = 5. So your 128x128 rects are now 128*5x128*5 = 640x640.
            Same goes for step_h and step_w
            Be careful!

        I include further notes for the UI's use in a README.pdf in this directory, read it!
    """
    """
    Open a new interactive session before starting our loops,
        and get the main canvas height and width from our session.
    We will use this for calculating our resize factor for each image later.
    """
    interactive_session = InteractiveGUI(dual_monitor)

    """
    Load the rects that our object detector detected for editing
    """
    with open(rects_archive_dir, "r") as f:
        rects = pickle.load(f)

    if not images_already_resized:
        print "Resizing Images..."
        with h5py.File(img_archive_dir, "r", chunks=True, compression="gzip") as img_hf:
            with h5py.File(resized_img_archive_dir, "w", chunks=True, compression="gzip") as resized_img_hf:
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
                    print "Resizing Image %i/%i..." % (img_i, img_n-1)

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
                    resized_img = cv2.resize(img, (0,0), fx=resize_factor, fy=resize_factor)

                    """
                    And write it to our resized img dataset.
                    """
                    resized_img_hf.create_dataset(str(img_i), data=resized_img)

    """
    Now that our resized image archive is ready, we open it for our images to use for displaying in our UI
    """
    print "Completed Resizing, starting UI..."
    with h5py.File(resized_img_archive_dir, "r", chunks=True, compression="gzip") as img_hf:
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
            print "Image %i/%i" % (img_i, img_n-1)

            """
            Get our image by resizing our block of memory instead of deleting and recreating it,
                and make sure to read directly from the dataset into this block of memory,
                so that we can reference it quickly when iterating through subsections for classification later.
            """
            img_dataset = img_hf.get(str(img_i))
            img.resize(img_dataset.shape)
            img_dataset.read_direct(img)

            """
            Resize our rects list via elementwise rects * resize_factor, 
                then make sure it's an int because we need to draw using these values (e.g. we can't have 1.5 pixels),
                and convert it to a list so we can append and delete from it easily in our interactive_gui_handler.py
            """
            rects[img_i] = rects[img_i] * resize_factor
            rects[img_i] = rects[img_i].astype(int)
            rects[img_i] = list(rects[img_i])

            """
            Set our interactive UI's img, rects, rect_h, and rect_w before starting the session and our main ui loop
            """
            interactive_session.np_img = img
            interactive_session.rects = rects[img_i]

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
            rects[img_i] = interactive_session.rects
            rects[img_i] = np.array(rects[img_i])
            rects[img_i] = rects[img_i] * (1/(resize_factor))
            rects[img_i] = rects[img_i].astype(int)

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
