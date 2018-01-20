from EditingDataset import *

class PredictionGrids(object):
    def __init__(self, dataset, uid):
        self.dataset = dataset#for reference, do not modify
        self.imgs = self.dataset.imgs
        self.uid = uid

        #Our two attributes for predictions before and after editing.
        self.archive_dir_before_editing = "../data/prediction_grids_before_editing/"#Where we'll store the .npy files for our predictions before editing
        self.archive_dir_after_editing = "../data/prediction_grids_after_editing/"#Where we'll store the .npy files for our predictions after editing
        self.before_editing = EditingDataset(self.dataset, self.uid, self.archive_dir_before_editing)
        self.after_editing = EditingDataset(self.dataset, self.uid, self.archive_dir_after_editing)

    def generate(self):
        pass

    def edit(self):
        pass
