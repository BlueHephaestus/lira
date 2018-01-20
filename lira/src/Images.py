import sys
import cv2
import numpy as np

from base import *

class Images(object):
    """
    Since this class references hard disk files at directories set
        in the class initialization,, reading and writing to any instance
        of this class will read and write the exact same images
        as other instances.
    """

    def __init__(self, restart=False):
        self.img_dir = "../../Input Images/"#where image files are stored
        self.archive_dir = "../data/images/"#where we will create and store the .npy archive files
        self.archives = []#where we will store list of full filepaths for each archive in our archive_dir
        self.max_shape = [0,0,0]#maximum dimensions of all images

        if restart:
            """
            Loop through all images in img_dir, create enumerated archives for them in archive_dir,
                and add each enumerated archive filepath to our archives list.
            """
            for i, fname in enumerate(fnames(self.img_dir)):
                #Progress indicator
                sys.stdout.write("\rArchiving Image {}/{}...".format(i, len([fname for fname in fnames(self.img_dir)])-1))

                #Read src, Check max shape, Create archive at dst, add dst to archive list
                src_fpath = os.path.join(self.img_dir, fname)
                dst_fpath = os.path.join(self.archive_dir, "{}.npy".format(i))
                img = cv2.imread(src_fpath)
                for i, dim in enumerate(img.shape):
                    if dim > self.max_shape[i]:
                        self.max_shape[i] = dim
                np.save(dst_fpath, img)
                self.archives.append(dst_fpath)

            sys.stdout.flush()
            print("")
        else:
            #use existing archive files
            for fname in fnames(self.archive_dir):
                self.archives.append(os.path.join(self.archive_dir, fname))

    def __iter__(self):
        for archive in self.archives:
            img = np.load(archive)
            yield img

    def __getitem__(self, i):
        return np.load(self.archives[i])

    def __setitem__(self, i, img):
        np.save(self.archives[i], img)

    def __len__(self):
        return len(self.archives)
