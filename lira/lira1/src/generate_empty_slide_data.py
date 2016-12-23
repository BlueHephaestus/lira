"""
Use this to generate n new 1160x640 images of the
    "Empty Slide" classification. Since we don't have
    any actual labeled empty slide images in the training data,
    we will be using a distribution with mean 244 and stddev 1
    to draw data from, as the slides are never perfectly one color,
    and average 244 from the sample I observed.

Of course, you can change these parameters to whatever you like.

-Blake Edwards / Dark Element
"""
import sys, os
import cv2
import numpy as np

#Name to save each slide by
save_fname = "empty_slides"

#Where to save our slides
save_dir = "../../data/raw_training/empty_slides"

#Our number of empty slides to generate
slide_n = 20

#Height
slide_h = 640

#Width
slide_w = 1160

#Mean 
slide_mean = 244

#Stddev
slide_stddev = 0.22

#Go through and remove any samples that are already there
for fname in os.listdir(save_dir):
    fpath = os.path.join(save_dir, fname)
    try: 
        if os.path.isfile(fpath):
            os.unlink(fpath)
    except:
        pass

"""
Get slide_n samples of our given shape from the normal distribution, 
    and then fit them to the mean and stddev specified
"""
slides = np.random.randn(slide_n, slide_h, slide_w)*slide_stddev + slide_mean

for slide_i, slide in enumerate(slides):
    sys.stdout.write("\rSlide #%i" % slide_i)
    sys.stdout.flush()

    #Write our slide
    cv2.imwrite("%s%i.jpg" % (os.path.join(save_dir, save_fname), slide_i), slide)
