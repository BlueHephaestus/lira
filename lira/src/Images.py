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

        if restart:
            """
            Loop through all images in img_dir, create enumerated archives for them in archive_dir,
                and add each enumerated archive filepath to our archives list.
            """
            #Delete all files in the archive directory if restarting
            clear_dir(self.archive_dir)

            for i, fname in enumerate(fnames(self.img_dir)):
                #Progress indicator
                sys.stdout.write("\rArchiving Image {}/{}...".format(i, len([fname for fname in fnames(self.img_dir)])-1))

                #Read src, Check max shape, Create archive at dst, add dst to archive list
                src_fpath = os.path.join(self.img_dir, fname)
                dst_fpath = os.path.join(self.archive_dir, "{}.npy".format(i))
                np.save(dst_fpath, cv2.imread(src_fpath))
                self.archives.append(dst_fpath)

            sys.stdout.flush()
            print("")
        else:
            #use existing archive files
            for fname in fnames(self.archive_dir):
                self.archives.append(os.path.join(self.archive_dir, fname))

        #Initialize to the original list of images ordered in the input images folder
        self.fnames = [fname for fname in fnames(self.img_dir)]

        #Regardless of this we sort the result, since it depends on the nondeterministic ordering of the os.walk generator in fnames()
        #We have to get the filename integer number, since otherwise we will end up with stuff like 0, 10, 11, 1 instead of 0, 1, 10, 11
        self.archives = sorted(self.archives, key=lambda x: int(x.split(os.sep)[-1][:-4]))

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

    def max_shape(self):
        max_shape = [0,0,0]#maximum dimensions of all images

        #load with mmap mode so we can just get the shape
        for archive in self.archives:
            img = np.load(archive, mmap_mode='r')
            for i, dim in enumerate(img.shape):
                if dim > max_shape[i]:
                    max_shape[i] = dim
        return max_shape
