import os
import numpy as np

from base import *

class EditingDataset(object):
    #For use with both predictions and detections, both before and after editing.
    def __init__(self, dataset, uid, archive_dir, restart=False):
        self.dataset = dataset#for reference, do not modify
        self.imgs = self.dataset.imgs
        self.uid = uid
        self.archive_dir = archive_dir
        self.restart = restart
        self.archives = []#list of full filepaths for each archive in archive_dir

        #Delete all files in the archive directory if restarting
        if self.restart:
            clear_dir(self.archive_dir, f=lambda d: "{}_img_".format(self.uid) in d)

        """
        Get index of each image in img_dir and create an archive filepath
            from this, then append to our self.archives list.
        """
        for i in range(len(self.imgs)):
            #We just use this for the image indices.
            fname = "{}_img_{}.npy".format(self.uid, i)

            #Get archive fpath and append
            fpath = os.path.join(self.archive_dir, fname)
            self.archives.append(fpath)

    def __iter__(self):
        for archive in self.archives:
            yield np.load(archive)

    def __getitem__(self, i):
        return np.load(self.archives[i])

    def __setitem__(self, i, data):
        return np.save(self.archives[i], data)

    def __len__(self):
        #Same as len(imgs) by definition
        return len(self.imgs)

