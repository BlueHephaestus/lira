"""
Since we do not have the memory available to run generate_overlay.py
    on the entire image at once, we divide the image into subsections 
    and classify those, before this.

This script takes those subsections and combines them into one big image again.

It works with both static divide factors, and dynamic, 
    i.e. some images aren't divided and have 1 final subsection, while some may be
        divided a bunch and have 64 total subsections, and so on.

So you can run it after generating the overlays and it should Just Work:tm:
"""

import sys, os, pickle, re
import numpy as np
import cv2

def remove_existing(existing_dir):
    #Go through and remove any samples that are already there
    for fname in os.listdir(existing_dir):
        fpath = os.path.join(existing_dir, fname)
        try: 
            if os.path.isfile(fpath):
                os.unlink(fpath)
        except:
            pass

def recursive_get_paths(img_dir):
    paths = []
    for (path, dirs, fnames) in os.walk(img_dir):
        for fname in fnames:
            paths.append((os.path.join(path, fname), fname))
    return paths

def get_img_ns(img_fnames):
    """
    Goes through ordered list of fnames and get the number of images that have the same index, 
        returning a list where the indices match up to show the number
    """
    img_ns = []
    for fname in img_fnames:
        #Get image index
        img_i = int(fname.split("_")[0])

        #Update number for this image index
        if len(img_ns) <= img_i:
            img_ns.append(1)
        else:
            img_ns[img_i] += 1

    return img_ns

#Note: Leave classification_metadata_dir as None to not add a classification key
def concatenate_results(results_dir, concatenated_results_dir, classification_metadata_dir=None):

    #Remove any existing files in our resulting directory
    remove_existing(concatenated_results_dir)
    
    #Get all our paths and fill list with only the paths
    overlay_sub_img_path_infos = recursive_get_paths(results_dir)
    overlay_sub_img_paths = [path for path, fname in overlay_sub_img_path_infos]

    """
    Then, we sort with priority order Image #, Row #, Col #.
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    overlay_sub_img_paths = sorted(overlay_sub_img_paths, key=alphanum_key)

    #Split on os seperation of file names in directory paths
    overlay_sub_imgs = [overlay_sub_img_path.split(os.sep)[-1] for overlay_sub_img_path in overlay_sub_img_paths]

    #Get the number of subsections for each image
    img_ns = get_img_ns(overlay_sub_imgs)
        
    #Prepare our image path storage list
    img_subs = [[] for img_i in img_ns]

    for overlay_sub_img in overlay_sub_imgs:
        #Loop through our images, and parse out the image #, row #, and col #
        img_i = int(overlay_sub_img.split("_")[0])

        #Append our sub img paths to a vector for each image so we can form the full matrix later
        img_subs[img_i].append(os.path.join(results_dir, overlay_sub_img))

    for img_i, img in enumerate(img_subs):
        sys.stdout.write("\rConcatenating Image #%i" % img_i)
        sys.stdout.flush()
        """
        We calculate our img_n, row_n, and col_n using our info for each img stored in img_ns list
            We assume our images are divided evenly into squares, e.g. 16 = 4x4, 64 = 8x8, etc.
        """
        img_n = img_ns[img_i]
        row_n = int(np.sqrt(img_n))
        col_n = row_n

        """
        Each img is a vector of the subsections, ordered left to right, top to bottom.
            So, we can convert it to a np array and reshape!
        """
        img = np.array(img).reshape((row_n, col_n))

        """
        Make a vector of lists, where each list is a row in the image.
            This way, we can go through and concatenate the row together,
            then append them, without knowing how big the image is.
        """
        result_rows = [[] for row_i in range(row_n)]
        for row_i, row in enumerate(img):
            result_cols = [[] for col_i in range(col_n)]
            for col_i, col in enumerate(row):
                result_cols[col_i] = cv2.imread(img[row_i][col_i])
            result_cols = np.concatenate([col for col in result_cols], axis=1)
            result_rows[row_i] = result_cols
        result_img = np.concatenate([row for row in result_rows], axis=0)

        if classification_metadata_dir:
            #Load our classifications and colors
            f = open(classification_metadata_dir, 'r')
            classification_metadata = pickle.load(f)
            classifications = classification_metadata[0]
            colors = classification_metadata[1]

            #With these, we loop through them in tandem and increment a key in the top right corner of our image
            #Note: we do 80x + 50 so we get the pattern of 50, 130, 210, etc to start at 50 and increment by 80
            for classification_i, classification in enumerate(classifications):
                cv2.putText(result_img, classifications[classification_i], (0, 80*classification_i + 50), cv2.FONT_HERSHEY_SIMPLEX, 2, colors[classification_i], 4)

        #Finally, write our result_img
        cv2.imwrite("%s/%i.jpg" % (concatenated_results_dir, img_i), result_img)
                
concatenate_results("results/", "concatenated_results/", classification_metadata_dir="classification_metadata.pkl")
