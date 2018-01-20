"""
Archives all images from our data_dir directory,
    into our archive_dir .h5 file,
    as either grayscales or rgb images, depending on the option set.

Usually, these images are quite large, so this program will take a while and use a lot of memory.

-Blake Edwards / Dark Element
"""

import os, sys
import numpy as np
import cv2
import h5py

sys.path.append(os.path.expanduser("~/programming/machine_learning/tuberculosis_project/lira_static/"))

import img_handler
from img_handler import *

def recursive_get_paths(img_dir):
    """
    Arguments:
        img_dir : directory to recursively traverse. Should be a string.

    Returns:
        A list of tuples, where each tuple is of the form
            (path and filename str, filename str), e.g.
            ("/home/darkelement/test.txt", "test.txt")
    """
    paths = []
    for (path, dirs, fnames) in os.walk(img_dir):
        for fname in fnames:
            paths.append((os.path.join(path, fname), fname))
    return paths

def create_archive(data_dir, archive_dir, rgb=False):
    """
    Arguments:
        data_dir: string directory where all image files are stored. 
            All files here should be images, that are readable by OpenCV
        archive_dir: string directory of a .h5 file to store images once loaded.
        rgb: Boolean for if we want to load rgb images (True), or grayscale images (False).
            Will use opencv's load to convert to grayscale if rgb=True, 
            however will not convert to rgb if images are grayscale, will instead just return the gray images.
            you should do this later when loading the images to save space.
            
    Returns:
        After iterating through each image, stores each image at the archive_dir location, 
            according to each image's index.
        Otherwise does not return anything.
    """
    sub_h = 80
    sub_w = 145

    print "Getting Image Paths..."
    sample_path_infos = recursive_get_paths(data_dir)

    """
    Open/Create our archive
    """
    print "Creating Archive..."
    with h5py.File(archive_dir, mode="w", chunks=True, compression="gzip") as hf:

        """
        Loop through samples
        """
        for sample_i, sample_path_info in enumerate(sample_path_infos):
            sample_path, sample_fname = sample_path_info

            #sys.stdout.write("\rArchiving Image #%i:%s" % (sample_i, sample_fname))
            #sys.stdout.flush()
            print"\rArchiving Image #%i:%s" % (sample_i, sample_fname)

            """
            Get appropriate version of img
            """
            if rgb:
                img = cv2.imread(sample_path)
            else:
                img = cv2.imread(sample_path, 0)

            """
            Archive img
            """
            hf.create_dataset(str(sample_i), data=img)

    print ""#flush formatting
