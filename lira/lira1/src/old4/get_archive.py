"""
This script goes through our samples_dir, which is formatted as such:
samples_dir/
    imgs/
        class_1
        ...
        class_n
    subs/
        class_1
        ...
        class_n
    live_samples.h5

Where:
    the imgs/ directory contains all the whole images,
    the subs/ directory contains all the individual subsections,
    and the live_samples.h5 contains all the samples we've obtained from our live system.

With this, it creates the final samples h5 archive in the archive_dir directory

TODO:
    Should we modify this to no longer use hard images, but only use our live_samples.h5?
    This would significantly shorten the code both here and in dataset_obj.py, if we choose to.
"""

import os, sys
import numpy as np
import cv2
import re
import json
import h5py

max_img_h = 640
max_img_w = 1160

#Pick ones that divide evenly!
sub_h = 80
sub_w = 145

def get_name(s):
    #Remove the numbering and extension
    s = re.sub("\d+", "", s)
    return s[:-4]

def get_label(path):
    #Get the path by separating on the separation character, and assigning the directory this file resides in as the label
    return path.split(os.sep)[-2]

def get_subsections(sub_h, sub_w, img):
    #Divide our greyscale into subsections of width sub_w and height sub_h
  
    #Set it to resulting size
    subs = np.zeros(shape=(img.shape[0]//sub_h, img.shape[1]//sub_w, sub_h, sub_w))#Final array of subsections

    #Use stride length to get our subsections to splice on
    sub_total = int(img.shape[0]//sub_h*img.shape[1]//sub_w)
    for row_i, row in enumerate(range(0, img.shape[0], sub_h)):
        for col_i, col in enumerate(range(0, img.shape[1], sub_w)):
            #Get the subsection specified by our loops
            sub = img[row:row+sub_h, col:col+sub_w]

            #Assign respective position
            subs[row_i, col_i] = sub
    return subs

def recursive_get_lens(whole_img_dir, individual_sub_dir, live_archive_dir):
    """
    Recursively get the number of files in whole_img_dir and individual_sub_dir, then get all the ones in live_archive_dir
    """

    """
    Once we get the number in whole_img_dir, we multiply by the number of resulting subsection images we will get for each image
    """
    whole_img_n = 0
    for (path, dirs, fnames) in os.walk(whole_img_dir):
        whole_img_n += len(fnames)

    subsection_n = int(max_img_h//sub_h * max_img_w//sub_w)
    whole_img_n *= subsection_n

    """
    Then we get individual subsection number
    """
    individual_sub_n = 0
    for (path, dirs, fnames) in os.walk(individual_sub_dir):
        individual_sub_n += len(fnames)

    """
    Finally, we get individual subsection number from our archive(if it exists)
    """
    try:
        with h5py.File(live_archive_dir,'r') as hf:
            individual_sub_n += len(np.array(hf.get("x")))
    except:
        pass

    sample_n = whole_img_n + individual_sub_n

    return sample_n, whole_img_n, individual_sub_n

def recursive_get_paths(img_dir):
    paths = []
    for (path, dirs, fnames) in os.walk(img_dir):
        for fname in fnames:
            paths.append((os.path.join(path, fname), fname))
    return paths

def get_archive(data_dir):

    """
    We go through the imgs/ directory, and we:
        1. get all of our whole images
        2. Divide them into subsections
        3. Add them to x and y positions in data
    We then go through the subs/ directory, and we:
        1. Add them to x and y positions in data
    Then we get live_archive.h5's individual subsections, and 
        1. Add them to x and y positions in data

    We keep track of the number of whole images and individual subsections for easy separation later.
    """

    whole_img_dir = os.path.join(data_dir, "imgs")
    individual_sub_dir = os.path.join(data_dir, "subs")
    live_archive_dir = os.path.join(data_dir, "live_samples.h5")


    #DENNIS takes vectors as inputs, so we convert our matrices accordingly
    """
    Get the number of total samples we will have, for both whole and subs
        Then we initialize our data array

        data =
                        X        Y
        |number     [      ], [label]
        |           [ img1 ]
        |           [      ]
        |of         [      ], [label]
        |           [ img2 ]
        |           [      ]
        |samples    [      ], [label]
        |           [ img3 ]
        |           [      ]
        v
    """

    print "Getting Number of Samples..."
    sample_n, whole_img_n, individual_sub_n = recursive_get_lens(whole_img_dir, individual_sub_dir, live_archive_dir)
    subsection_n = int(max_img_h//sub_h * max_img_w//sub_w)
    data = [np.zeros(shape=(sample_n, sub_h*sub_w), dtype=np.float32), np.zeros(shape=(sample_n), dtype=np.int)]
    """
    Get all the paths and filenames for our images in both, so that each element is (full path, filename) 
    """
    print "Getting Sample Paths..."
    whole_img_path_infos = recursive_get_paths(whole_img_dir)
    individual_sub_path_infos = recursive_get_paths(individual_sub_dir)

    #Our dictionary of each label and it's number
    label_dict = {}
    
    #Our number of samples to increment as we go through each
    sample_i = 0

    #Our number of labels to increment as we go through each
    label_i = 0
    """
    Now, Handle whole imgs first.
    """
    print "Getting Whole Image Samples..."
    for whole_img_path_info in whole_img_path_infos:
        #Get our path and fname from the tuple stored
        sample_path, sample_fname = whole_img_path_info

        #Get our label via the file it resides in
        sample_label = get_label(sample_path)

        #Unfortunately we have to keep label_i and label_num separate so we don't get incorrect values when we find a new class.
        if sample_label not in label_dict:
            label_dict[sample_label] = label_i
            label_i+=1
        label_num = label_dict[sample_label] 

        #Get greyscale version of img
        img = cv2.imread(sample_path, 0)

        #Resize accordingly.
        #Healthy tissue is the only one smaller than this, and only by ~20 pixels. 
        #The loss should not make any difference whatsoever as those slides are nearly homogenous
        img = np.resize(img, (max_img_h, max_img_w))

        #Divide into subsections 
        subs = get_subsections(sub_h, sub_w, img)
        for row in subs:
            for sub in row:
                #Flatten so we have our input vector format
                data[0][sample_i] = sub.flatten()
                data[1][sample_i] = label_num

                sample_i+=1

    """
    Then, handle individual subs in both directory and archive
    First, directory
    """
    print "Getting Individual Subsection Samples from Directory..."
    for individual_sub_path_info in individual_sub_path_infos:
        #Get our path and fname from the tuple stored
        sample_path, sample_fname = individual_sub_path_info

        #Get our label via the file it resides in
        sample_label = get_label(sample_path)

        #Unfortunately we have to keep label_i and label_num separate so we don't get incorrect values when we find a new class.
        if sample_label not in label_dict:
            label_dict[sample_label] = label_i
            label_i+=1
        label_num = label_dict[sample_label] 

        #Get greyscale version of img
        img = cv2.imread(sample_path, 0)

        #If this image is 145x80 instead of 80x145, rotate 90 degrees.
        if img.shape[0] > img.shape[1]:
            img = np.rot90(img)

        data[0][sample_i] = img.flatten()
        data[1][sample_i] = label_num

        sample_i+=1
    """
    Now handle the ones in the archive (if it exists)
    """
    print "Getting Individual Subsection Samples from Live Archive..."
    try:
        #We add these to the remaining locations
        with h5py.File(live_archive_dir,'r') as hf:
            data[0][sample_i:] = np.array(hf.get("x"))
            data[1][sample_i:] = np.array(hf.get("y"))
    except:
        pass

    #Save this as json so we can store it with h5py
    label_dict = json.dumps(label_dict)

    print "\nNumber of samples: %i\nNumber of whole images: %i\nNumber of individual subsections: %i" % (sample_n, whole_img_n, individual_sub_n)
    print "Final Label Dict: {}\n".format(label_dict)
    metadata = [subsection_n, sample_n, whole_img_n, individual_sub_n, label_dict]
    return data, metadata

def regenerate_archive(archive_dir="../data/samples.h5", data_dir="../data/samples"):
    data, metadata = get_archive(data_dir)

    #Store our data as h5 archive
    print "Archiving Data..."

    with h5py.File(archive_dir, 'w') as hf:
        hf.create_dataset("x", data=data[0])
        hf.create_dataset("y", data=data[1])
        hf.create_dataset("metadata", data=metadata)

#regenerate_archive()
