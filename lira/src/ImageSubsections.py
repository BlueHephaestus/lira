import numpy as np

class ImageSubsections(object):
    """
    Given an image and a subsection height and width, this class 
        acts as an interface for the image as if it were a vector
        of subsections of this size, without actually dividing the image
        as doing so would require too much storage.
    """
    def __init__(self, img, sub_h, sub_w):
        self.img = img
        self.sub_h = sub_h
        self.sub_w = sub_w

    def __iter__(self):
        #Loop through subsections of size sub_hxsub_w in our image.
        for row_i in range(0, self.img.shape[0]-self.sub_h, self.sub_h):
            for col_i in range(0, self.img.shape[1]-self.sub_w, self.sub_w):
                yield self.img[row_i:row_i+self.sub_h, col_i:col_i+self.sub_w]

    def __getitem__(self, indices):
        #Handle indexing with lists only, since that's the only type of indexing we use with this class.
        #We return a 4d np array 
        row_indices = indices // (self.img.shape[1]//self.sub_w)
        col_indices = indices % (self.img.shape[1]//self.sub_w)

        #We scale them up from subsection resolution to match our image resolution now that we have the 2d cords.
        row_indices = row_indices * self.sub_h
        col_indices = col_indices * self.sub_w

        return np.array([self.img[row_i:row_i+self.sub_h, col_i:col_i+self.sub_w] for (row_i, col_i) in zip(row_indices, col_indices)])

    def __len__(self):
        return (self.img.shape[0]//self.sub_h)*(self.img.shape[1]//self.sub_w)

