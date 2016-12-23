"""
Archives svs greyscales

svs files vary in size wildly, always at least 20,000 x 20,000

-Blake Edwards / Dark Element
"""

import os
import cv2#opencv
import pickle, gzip

#For enumeration

def disp_image(img):
    #Disp the image and close on first keystroke
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_greyscales(data_dir, archive_dir):
    #Loop through all the samples in our data_dir,
    #Appending to our greyscales array
    
    print "Loading greyscales..."

    f = gzip.open(archive_dir, "wb")

    #greyscales = []
    sample_num = 1
    for sample in os.listdir(os.path.abspath(data_dir)):
      if ".svs" in sample:
        #Get full path of .svs
        sample_path = data_dir + "/" + sample

        print "\tGetting greyscale of sample #%i:%s" % (sample_num, sample)

        #Get greyscale version of img
        img = cv2.imread(sample_path, 0)

        #Archive greyscale
        pickle.dump((img), f, protocol=-1)

        sample_num += 1
        #greyscales.append(img)

    f.close()
    
#load_greyscales("../data", "../data/greyscales.pkl.gz")
