"""
Generate an h5py file for our X and Y labelled data to use for training, and testing.
    (not doing validation dataset due to the small number of optimizations on hyperparameters)
We do this by looping through our negative and positive dirs, and setting the Y label as 0 if negative, and 1 if positive

This is a really small script, and I may only use it once due to the extremely problem-specific nature of it.

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

"""
Directories to get our negative and positive samples from
"""
negatives_dir = "data/negatives/"
positives_dir = "data/positives/"

"""
H5py file directory to store samples
"""
archive_dir = "samples.h5"

"""
Dimensions of one x sample, 
    and if we have our samples as rgb or not
"""
sample_dims = [128, 128, 3]
rgb = True

"""
Get our filepaths
"""
negative_fpaths = recursive_get_paths(negatives_dir)
positive_fpaths = recursive_get_paths(positives_dir)

"""
Get the total number of samples via the total number of filepaths
"""
sample_n = len(negative_fpaths) + len(positive_fpaths)

"""
Create our x so that we can directly insert each image into it, we will wait to reshape it until they have all been added.
"""
X_dims = [sample_n]
X_dims.extend(sample_dims)

"""
Initialize X and Y with known data numbers
"""
#X = np.zeros(X_dims)
X = np.memmap("x.dat", dtype="uint8", mode="r+", shape=tuple(X_dims))

Y = np.zeros(sample_n,)

print "Getting Negative Samples..."
for i, negative_fpath_info in enumerate(negative_fpaths):
    """
    Loop through our paths and get our negative samples, 
        setting their y value as 0
    """
    negative_fpath, negative_fname = negative_fpath_info
    if rgb:
        X[i] = cv2.imread(negative_fpath)
    else:
        X[i] = cv2.imread(negative_fpath, 0)

    Y[i] = 0

print "Getting Positive Samples..."
for i, positive_fpath_info in enumerate(positive_fpaths):
    """
    Loop through our paths and get our positive samples, 
        setting their y value as 1
    """
    positive_fpath, positive_fname = positive_fpath_info
    if rgb:
        X[i] = cv2.imread(positive_fpath)
    else:
        X[i] = cv2.imread(positive_fpath, 0)
    Y[i] = 1

"""
Reshape accordingly, flattening our h and w of our image
"""
if rgb:
    X = np.reshape(X, (sample_n, -1, 3))
else:
    X = np.reshape(X, (sample_n, -1))

"""
Create our archive
"""
print "Creating Archive..."
with h5py.File(archive_dir, "w", chunks=True, compression="gzip") as hf:
    hf.create_dataset("x", data=X)
    hf.create_dataset("x_shape", data=X.shape)
    hf.create_dataset("y", data=Y)
    hf.create_dataset("y_shape", data=Y.shape)

