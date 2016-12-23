import numpy as np
import cv2
import pickle, gzip, sys

import svs_loader

data_dir = "../data"
#subsection_dir = "../data/subsections"
subsection_dir = "../data/256subsections"
archive_dir = "../data/greyscales.pkl.gz"
subsection_archive_dir = "../data/greyscale_subs.pkl.gz"

#For image subsections
#sub_w = 28
#sub_h = 28
sub_w = 256
sub_h = 256

sub_top_intensities_num = 10

def get_nd_len(a):
  #Convert to np ndarray
  a = np.array(a)

  #Get the product of the shape, i.e. (480, 640) = 480*640
  l = np.prod(a.shape)
  return l

def get_subsections(sub_w, sub_h, img, img_i, f):
    #Divide our greyscale into subsections of width sub_w and height sub_h
  
    #Set it to resulting size
    #subs = np.zeros(shape=(img.shape[0]//sub_h, img.shape[1]//sub_w, sub_h, sub_w))#Final array of subsections

    #Use stride length to get our subsections to splice on
    sub_i = 0
    sub_total = int(img.shape[0]//sub_h*img.shape[1]//sub_w)
    for row in range(0, img.shape[0], sub_h):
        for col in range(0, img.shape[1], sub_w):
            #Get the subsection specified by our loops
            sub = img[row:row+sub_h, col:col+sub_w]

            #Get the subsection-relative numbers
            row_i = row/sub_h
            col_i = col/sub_w

            sys.stdout.write("\r\t\tSubsection #%i / %i" % (sub_i, sub_total))
            sys.stdout.flush()

            """
            #Add the sub to our archive with metadata
            img_metadata = (greyscale_i, row_i, col_i)
            #pickle.dump((img_metadata, img), f, protocol=-1)
            pickle.dump((img_metadata, img), f)
            """
            #Write it to a png
            greyscale_sub_fname = "%s/%s_%i_%i.png" % (subsection_dir, greyscale_i, row_i, col_i)
            cv2.imwrite(greyscale_sub_fname, sub)

            sub_i += 1
    print ""#For flush print formatting

greyscale_f = gzip.open(archive_dir, 'rb')
greyscale_subs_f = gzip.open(subsection_archive_dir, "wb")

greyscale_i = 0
#Keep unpickling until we reach EOF and error out.
try:
    while True:
        print "\tGetting Subsections of Sample #%i" % (greyscale_i)

        #Unpickle greyscale
        greyscale = pickle.load(greyscale_f)

        #Get greyscale subsections
        get_subsections(sub_w, sub_h, greyscale, greyscale_i, greyscale_subs_f)
        greyscale_i += 1

except:
    print "DONE"
    pass
