
import os, sys
import numpy as np
import cv2
#import pickle, gzip
import re
import h5py

max_img_h = 640
max_img_w = 1160

#Pick ones that divide evenly!
#sub_h = 160
#sub_w = 290
sub_h = 80
sub_w = 145

def get_name(s):
    #Remove the numbering
    s = re.sub("\d+", "", s)
    return s[:-4]

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

def get_archive(data_dir, rim_subfolder, subsections=False):

    """
    Determine the number of normal samples (big images) and our rim images (individual subsections)
        for use in determining the dimensions of our data array
    """

    sample_total = len(os.listdir(os.path.abspath(data_dir)))
    rim_sample_total = len(os.listdir(os.path.abspath("%s/%s" % (data_dir, rim_subfolder))))

    #My pre-existing program takes vectors of values as inputs, not matrices
    #They are basically equivalent in this case.
    if subsections:
        sample_total = sample_total * (max_img_h//sub_h * max_img_w//sub_w) + rim_sample_total
        data = [np.zeros(shape=(sample_total, sub_h*sub_w), dtype=np.float32), np.zeros(shape=(sample_total), dtype=np.int)]
    else:
        sample_total = sample_total + rim_sample_total
        data = [np.zeros(shape=(sample_total, max_img_h*max_img_w), dtype=np.float32), np.zeros(shape=(sample_total), dtype=np.int)]

    '''
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
    '''

    print "Getting Pixel Data..."
    label_total = 0#Increments so that each sample has it's own label
    label_dict = {}#Keep track of sample:label numbers 
    sample_num = 0

    for sample in os.listdir(os.path.abspath(data_dir)):

        #If this is our raw rim directory, handle each file accordingly 
        #Each file is already a 80x145 image
        if rim_subfolder in sample:
            #Get full directory
            rim_image_dir =  "%s/%s" % (data_dir, sample)

            #Get all the filenames in the full directory
            rim_image_fnames = os.listdir(rim_image_dir)

            #Get all the full filepaths by combining these two
            rim_image_fpaths = ["%s/%s" % (rim_image_dir, rim_image_fname) for rim_image_fname in rim_image_fnames]

            #For our label dict
            rim_label = "Type I Rim"

            for rim_image_i, rim_image_fpath in enumerate(rim_image_fpaths):

                #Increment our total # of rim images and put this in the label dict
                if rim_label not in label_dict:
                    label_dict[rim_label] = label_total
                    label_total+=1
                label_num = label_dict[rim_label]

                #Load the img
                img = cv2.imread(rim_image_fpath, 0)

                #If this image is 145x80 instead of 80x145, rotate 90 degrees.
                if img.shape[0] > img.shape[1]:
                    img = np.rot90(img)

                data[0][sample_num] = img.flatten()
                data[1][sample_num] = label_num

                sample_num+=1

        else:
            #This is not our raw_rim directory, handle our entire images as we would normally
            input_fname = data_dir + "/" + sample
            sample = get_name(sample)

            if sample not in label_dict:
                label_dict[sample] = label_total
                label_total+=1
            label_num = label_dict[sample] 

            #Get greyscale version of img
            img = cv2.imread(input_fname, 0)

            #Resize accordingly.
            #Healthy tissue is the only one smaller than this, and only by ~20 pixels. 
            #The loss should not make any difference whatsoever as those slides are nearly homogenous
            img = np.resize(img, (max_img_h, max_img_w))

            if subsections:
                #Divide into subsections 
                subs = get_subsections(sub_h, sub_w, img)
                for row in subs:
                    for sub in row:
                        data[0][sample_num] = sub.flatten()#So we can use it as input vector
                        data[1][sample_num] = label_num

                        sample_num+=1
            else:
                #Return whole image
                data[0][sample_num] = img.flatten()#So we can use it as input vector
                data[1][sample_num] = label_num

                sample_num+=1

        sys.stdout.write("\rSample #%i" % sample_num)
        sys.stdout.flush()


    print "\nFinal Label Dict: {}".format(label_dict)
    return data

def regenerate_archive(archive_dir="../../data/samples.h5", data_dir="../../data/samples", rim_subfolder = "raw_rim",  subsections=True):
    data = get_archive(data_dir, rim_subfolder,  subsections=subsections)

    print "Archiving Data..."
    """
    f = gzip.open(archive_dir, "wb")
    pickle.dump((data), f, protocol=-1)
    f.close()
    """

    with h5py.File(archive_dir, 'w') as hf:
        hf.create_dataset("x", data=data[0])
        hf.create_dataset("y", data=data[1])

regenerate_archive()
