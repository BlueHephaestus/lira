"""
For easily getting a dataset object with the methods and objects we need to access easily in training.

-Blake Edwards / Dark Element
"""

import sys, os, gzip, cPickle, json
import h5py
import numpy as np

def get_one_hot_m(v, width):
    #Given vector and width of each one hot, 
    #   get one hot matrix such that each index specified becomes a row in the matrix
    m = np.zeros(shape=(len(v), width))
    m[np.arange(len(v)), v] = 1
    return m

def unison_shuffle(a, b):
    #Shuffle our two arrays while retaining the relation between them
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def whole_normalize_data(data, mean, std):
    data[0] = data[0]*std + mean
    return data

def get_data_subsets(archive_dir, p_training=0.8, p_validation=0.1, p_test=0.1):
    """
    Get our dataset array and seperate it into training, validation, and test data
        according to the percentages passed in.

    Defaults:
        80% - Training
        10% - Validation
        10% - Test

    This one is our general function, which will work unless we have a strange dataset bias situation as in the LIRA example.
    """

    print "Getting Training, Validation, and Test Data..."
    with h5py.File(archive_dir,'r') as hf:
        data = [np.array(hf.get("x")), np.array(hf.get("y"))]

    n_samples = len(data[0])

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
    unison_shuffle(data[0], data[1])

    #Get actual subsets
    data_x_subsets = np.split(data[0], [n_training_subset, n_training_subset+n_validation_subset])#basically the lines we cut to get our 3 subsections
    data_y_subsets = np.split(data[1], [n_training_subset, n_training_subset+n_validation_subset])

    training_data[0] = data_x_subsets[0]
    validation_data[0] = data_x_subsets[1]
    test_data[0] = data_x_subsets[2]

    training_data[1] = data_y_subsets[0]
    validation_data[1] = data_y_subsets[1]
    test_data[1] = data_y_subsets[2]

    return training_data, validation_data, test_data

def load_dataset_obj(p_training, p_validation, p_test, archive_dir, output_dims, whole_data_normalization=True):
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
    training_data[1] = get_one_hot_m(training_data[1], output_dims)
    validation_data[1] = get_one_hot_m(validation_data[1], output_dims)
    test_data[1] = get_one_hot_m(test_data[1], output_dims)

    #return dataset obj
    return Dataset(training_data, validation_data, test_data), whole_normalization_data

class Dataset(object):

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

