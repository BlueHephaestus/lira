import os, sys
import numpy as np
import cv2

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

def load_images(img_dir):
    """
    Arguments:
        img_dir: string directory where all image files are stored. 
            All files here should be images, that are readable by OpenCV.

        We furthermore assume these images are RGB images,
            and that each pixel is in the range of 0-255, so that they are of
            type uint8.
            
    Returns:
        Iterates through each image and adds them all to a numpy memmap, which is returned.
    """
    print "Getting all Images from Input Directory..."
    path_infos = recursive_get_paths(img_dir)

    #Create memmap
    imgs = np.memmap("../data/images.dat", dtype="uint8", mode="w", shape=(
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
