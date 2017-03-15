"""
Archives all images as greyscales from our data_dir directory,
    into our archive_dir .h5 file.

Usually, these images are quite large, so this program will take a while and use a lot of memory.

-Blake Edwards / Dark Element
"""

import os, sys
import numpy as np
import cv2
import h5py

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

def load_greyscales(data_dir, archive_dir):
    """
    Arguments:
        data_dir: string directory where all greyscale image files are stored. 
            All files here should be images, that are readable by OpenCV
        archive_dir: string directory of a .h5 file to store greyscale images once loaded.
    
    Returns:
        After iterating through each image, stores each image at the archive_dir location, 
            according to each image's index.
        Otherwise does not return anything.
    """

    print "Getting Image Paths..."
    sample_path_infos = recursive_get_paths(data_dir)

    #Open our archive
    print "Creating Archive..."
    with h5py.File(archive_dir, "w") as hf:

        #Loop through samples
        for sample_i, sample_path_info in enumerate(sample_path_infos):
            sample_path, sample_fname = sample_path_info

            sys.stdout.write("\rGetting Greyscale of Image #%i:%s" % (sample_i, sample_fname))
            sys.stdout.flush()

            #Get greyscale version of img
            img = cv2.imread(sample_path, 0)

            #Archive greyscale
            hf.create_dataset(str(sample_i), data=img)

    print ""#flush formatting

#load_greyscales("../data/test_slides", "../data/greyscales.h5")
