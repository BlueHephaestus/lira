
import os, sys
import numpy as np
import cv2
import pickle, gzip, re

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

def get_archive(data_dir, subsections=False):

    #Determine the dims of our big data array
    #Am I a data scientist nao XD
    sample_total = len(os.listdir(os.path.abspath(data_dir)))
    
    #My pre-existing program takes vectors of values as inputs, not matrices
    #They are basically equivalent in this case.
    if subsections:
        data = [np.zeros(shape=(sample_total * (max_img_h//sub_h * max_img_w//sub_w), sub_h*sub_w), dtype=np.float32), np.zeros(shape=(sample_total * (max_img_h//sub_h * max_img_w//sub_w)), dtype=np.int)]
    else:
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

        input_fname = data_dir + "/" + sample
        sample = get_name(sample)

        if sample not in label_dict:
            label_dict[sample] = label_total
            label_total+=1
        label_num = label_dict[sample] 

        #Get greyscale version of img
        img = cv2.imread(input_fname, 0)

        #Resize accordingly.
        #Healty tissue is the only one smaller than this, and only by ~20 pixels. 
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

def regenerate_archive(archive_dir="../data/samples_subs2.pkl.gz", data_dir="../data/samples", subsections=True):
    data = get_archive(data_dir, subsections=subsections)

    print data
    print data[0]
    print data[0][0]
    print data[0][0][0]

    print "Archiving Data..."
    f = gzip.open(archive_dir, "wb")
    pickle.dump((data), f, protocol=-1)
    f.close()

regenerate_archive()
