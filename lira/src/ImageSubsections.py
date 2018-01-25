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
                yield img[row_i:row_i+sub_h, col_i:col_i+sub_w]

    def __getitem__(self, i):
        row_i = i // self.img.shape[1]
        col_i = i % self.img.shape[1]
        return img[row_i:row_i+sub_h, col_i:col_i+sub_w]

    def __setitem__(self, i, data):
        row_i = i // self.img.shape[1]
        col_i = i % self.img.shape[1]
        img[row_i:row_i+sub_h, col_i:col_i+sub_w] = data

    def __len__(self):
        return (self.img.shape[0]//self.sub_h)*(self.img.shape[1]//self.sub_w)

