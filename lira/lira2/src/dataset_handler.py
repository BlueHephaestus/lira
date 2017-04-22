"""
For easily getting a dataset object with the methods and objects we need to access easily in training a model.

As many of the methods vary in their functionality, they each have their own documentation.
-Blake Edwards / Dark Element
"""

import sys, os, gzip, cPickle, json
import h5py
import numpy as np
from keras.utils import np_utils

def unison_shuffle(a, b):
    """
    Arguments:
        a, b: np arrays which share the same first dimension shape, 
            e.g. a -> (60, 28, 28), b -> (60,)
    
    Returns:
        Shuffles the two np arrays given in the exact same way, so that an element stored at an index in a with it's matching pair or label in the same index in b
            will retain that relation even after this shuffle.
        E.g. Before shuffle: a[4] = "some data", b[4] = 3
             After shuffle:  a[random_new_index] = "some data", b[random_new_index] = 3
        Does not return any values.
    """
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def whole_normalize_data(data, mean, std):
    """
    Arguments:
        data: np array with data[0] as our input data, and data[1] as output data
        mean: mean to normalize our data with. Set to 0 for no change.
        std: std to normalize our data with. Set to 1 for no change.
    Returns:
        Whole normalize our data to the standard normal distribution, via
            x-mean
            ------
             std
        Which we get from basic gaussian distribution statistics.
        If you don't want to normalize, just set mean = 0 and std = 1, which results in 
            x - 0
        y = -----
              1
        y = x
    """
    data[0] = (data[0]-mean)/std
    return data

def get_data_subsets(archive_dir, p_training=0.8, p_validation=0.1, p_test=0.1):
    """
    Arguments:
        archive_dir: String where .h5 file is stored containing model's data.
        p_training, p_validation, p_test: Percentage distribution of entire dataset to give to each subset.
            These should sum to 1.
            Defaults:
                .8 / 80% - Training
                .1 / 10% - Validation
                .1 / 10% - Test
        
    Returns:
        Loads the data stored in our archive_dir.
            This dataset should contain two np arrays, x and y, of shapes 
                x: (sample_n, ...)
                y: (sample_n,)
        From this data, and the number of samples, we divide the data by the proportions given 
            into each data subset, and return these three subsets (training, validation, and test)
    """

    print "Getting Training, Validation, and Test Data..."
    with h5py.File(archive_dir, "r", chunks=True, compression="gzip") as hf:
        x_shape = tuple(hf.get("x_shape"))
        y_shape = tuple(hf.get("y_shape"))
        x = np.memmap("x.dat", dtype="float32", mode="w+", shape=x_shape)
        y = np.memmap("y.dat", dtype="float32", mode="w+", shape=y_shape)
        x[:] = hf.get("x")[:]
        y[:] = hf.get("y")[:]

    n_samples = len(x)

    #Now we split our samples according to percentage
    training_data = [[], []]
    validation_data = [[], []]
    test_data = [[], []]

    n_training_subset = int(np.floor(p_training*n_samples))
    n_validation_subset = int(np.floor(p_validation*n_samples))
    #Assign this to it's respective percentage and whatever is left
    n_test_subset = n_samples - n_training_subset - n_validation_subset

    #Shuffle while retaining element correspondence
    print "Shuffling data..."
    unison_shuffle(x, y)

    #Get actual subsets
    data_x_subsets = np.split(x, [n_training_subset, n_training_subset+n_validation_subset])#basically the lines we cut to get our 3 subsections
    data_y_subsets = np.split(y, [n_training_subset, n_training_subset+n_validation_subset])

    training_data[0] = data_x_subsets[0]
    validation_data[0] = data_x_subsets[1]
    test_data[0] = data_x_subsets[2]

    training_data[1] = data_y_subsets[0]
    validation_data[1] = data_y_subsets[1]
    test_data[1] = data_y_subsets[2]

    return training_data, validation_data, test_data

def load_dataset_obj(p_training, p_validation, p_test, archive_dir, output_dims, whole_data_normalization=True):
    """
    Arguments:
        p_training, p_validation, p_test: Percentage distribution of entire dataset to give to each subset.
            These should sum to 1.
            Defaults:
                .8 / 80% - Training
                .1 / 10% - Validation
                .1 / 10% - Test
        archive_dir: String where .h5 file is stored containing model's data.
        output_dims: int for the number of output dimensions, or classes.
        whole_data_normalization: boolean representing whether to mean & stddev normalize our data or not.

    Returns:
        Gets the data subsets from `archive_dir`, 
        Normalizes them with mean & stddev normalization if enabled, 
        Converts the y labels into one-hots,
        then creates a Dataset object for easy reference of the data subsets,
        and returns this Dataset object.
    """
    training_data, validation_data, test_data = get_data_subsets(archive_dir, p_training=p_training, p_validation=p_validation, p_test=p_test)
        
    #Do whole data normalization on our input data, by getting the mean and stddev of the training data,
    #Then keeping these metrics and applying to the other data subsets
    if whole_data_normalization:
        training_data_mean, training_data_std = np.mean(training_data[0]), np.std(training_data[0])
        training_data =     whole_normalize_data(training_data, training_data_mean, training_data_std)
        validation_data =   whole_normalize_data(validation_data, training_data_mean, training_data_std)
        test_data =         whole_normalize_data(test_data, training_data_mean, training_data_std)
    else:
        training_data_mean = 0.0
        training_data_std = 1.0
    whole_normalization_data = [training_data_mean, training_data_std]

    #Convert ys in each to one hot vectors
    training_data[1] = np_utils.to_categorical(training_data[1], output_dims)
    validation_data[1] = np_utils.to_categorical(validation_data[1], output_dims)
    test_data[1] = np_utils.to_categorical(test_data[1], output_dims)

    return Dataset(training_data, validation_data, test_data), whole_normalization_data

class Dataset(object):
    """
    Our object for easy reference of data subsets and their x and y component parts.
        when properly initialized, can be referenced via    
            self.training -> training data DataSubset object
            self.validation -> validation data DataSubset object
            self.test -> test data DataSubset object
            self.training.x -> input training data
            self.training.y -> output training data
            ||
            \/ and so on...
        Also has next_batch method for when training on one mini batch each epoch,
            where the training dataset is shuffled and a mini batch subsection of it returned.
    """

    #Initialize our data subset objects
    def __init__(self, training_data, validation_data, test_data):
        self.training = training_subset(training_data)
        self.validation = validation_subset(validation_data)
        self.test = test_subset(test_data)

    #Get a new mb_n number of entries from our training subset, after shuffling both sides in unison
    def next_batch(self, mb_n):
        #Shuffle our training dataset,
        #Return first mb_n elements of shuffled dataset
        unison_shuffle(self.training.x, self.training.y)
        return [self.training.x[:mb_n], self.training.y[:mb_n]]

#So we assure we have the same attributes for each subset
class DataSubset(object):
    def __init__(self, data):
        #self.whole_data = data
        self.x = data[0]
        self.y = data[1]

class training_subset(DataSubset):
    def __init__(self, data):
        DataSubset.__init__(self, data)

class validation_subset(DataSubset):
    def __init__(self, data):
        DataSubset.__init__(self, data)

class test_subset(DataSubset):
    def __init__(self, data):
        DataSubset.__init__(self, data)

