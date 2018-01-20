"""
The main function for this file is generate_detected_objects(), so the bulk of the documentation for this file lies there.

-Blake Edwards / Dark Element
"""
import numpy as np
import pickle
import h5py

import object_detection_handler
from object_detection_handler import *

def generate_detected_objects(object_detection_model, model_dir="../lira/lira2/saved_networks", img_archive_dir="../lira/lira1/data/images.h5", rects_archive_dir="../lira/lira1/data/bounding_rects.pkl"):
    """
    Arguments:
        object_detection_model: String containing the filename of where our detection model is stored, to be used for detecting Type 1 Classifications in our images, 
            and drawing bounding rectangles on them.
        model_dir: Directory of where our model is stored. Will be used `object_detection_model` to obtain where the model is now located 
        img_archive_dir: a string filepath of the .h5 file where the images / greyscales are stored.
        rects_archive_dir: a string filepath of the .pkl file where the detected rects will be stored.
    Returns:
        Provided an image archive in img_archive_dir, 
            loop through all images in that archive, and for each image:
                get their detected objects with our object detection model loaded from model_dir and object_detection_model,
                and add each image's detected objects to a list of `rects`,
                    since these detected objects will be rectangles.
            Once done looping through all images and building our rects list, 
                we write these rects to a new .pkl file.
    """
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
        Initialize our final rects list, which we will build with detected rects from each image.
        """
        rects = []

        """
        Start looping through images
        """
        for img_i in range(img_n):
            """
            Get our image by resizing our block of memory instead of deleting and recreating it,
                and make sure to read directly from the dataset into this block of memory,
                so that we can reference it quickly when iterating through subsections for classification later.
            """
            img_dataset = img_hf.get(str(img_i))
            img.resize(img_dataset.shape)
            img_dataset.read_direct(img)

            """
            Open our saved object detection model
            """
            object_detector = ObjectDetector(object_detection_model, model_dir)

            """
            Get all bounding rectangles for type 1 classifications using our detection model, on our entire image.
                (parameters for this model are defined in the object_detection_handler.py file, because this is a very problem-specific addition to this file)
            """
            img_rects = object_detector.generate_bounding_rects(img)

            """
            Now that we have our img_rects (which is a np array, btw), add them to the rects with a simple append.
            """
            rects.append(img_rects)
            if img_i >= 26:
                break

    """
    Finally write all our detected rects to our rects_archive_dir
    """
    with open(rects_archive_dir, "w") as f:
        pickle.dump(rects, f)

