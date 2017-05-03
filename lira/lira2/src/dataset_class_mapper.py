"""
If you are wanting to train a network with only a few of the LIRA classes, e.g. with this distribution:
    0: 12159
    1: 6055
    2: 0
    3: 7000
    4: 0
    5: 8972
    6: 0

Where the left are the class indices and the right are the sample numbers.

You will run into a problem, as the indices will still be stored as [0, 1, 3, 5].
This means that if I try to convert the y vector/matrix into a categorical matrix/tensor,
    with the width = 4 (since there are only 4 classes in this example),
When I run into a sample of index 5 the following will happen:


    width = 4:

    [0, 0, 0, 0] and 1 -> [0, 1, 0, 0]
    [0, 0, 0, 0] and 3 -> [0, 0, 0, 1]
    [0, 0, 0, 0] and 5 -> Index out of bounds

So we'd like a quick way to modify the index counts in order to have them be 

    [0, 1, 3, 5] -> [0, 1, 2, 3]

To reflect the change in the samples we are training on.

This is that quick way. Change the dictionary mapper argument to reflect the class changes

-Blake Edwards / Dark Element
"""
import h5py
import numpy as np

def rename_classes(archive_dir, class_mapper): 
    """
    Arguments
        archive_dir: String where .h5 file is stored containing model's data.
        class_mapper: Dictionary/Hashmap for each index that will be encountered in the model's y data vector / labels,
            and it's associated new index.

    Returns:
        Returns no value, but modifies the y values in archive_dir to reflect their new indices.
        Use with caution, you may mess up your labels if you aren't careful.
    """
    """
    Get our predictions, and leave file open since we are just doing a quick loop
    """
    with h5py.File(archive_dir, "r+") as hf:
        """
        We don't make this a np.array, so we can modify the file contents while we loop
        """
        y = hf.get("y")

        """
        Loop through our predictions,
            and set them to the mapped new prediction indices 

        This is all we need to do, since we are modifying the contents of the file with each assignment in this loop
        """
        for i in range(len(y)):
            y[i] = class_mapper[y[i]]


#rename_classes("../data/model_1_samples.h5", {0:0, 1:1, 3:2, 5:3})
#rename_classes("../data/model_2_samples.h5", {0:0, 2:1, 3:2, 4:3})
