import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.regularizers import l2
from keras import applications

import dataset_handler
from dataset_handler import *

"""
Data Parameters
"""
p_training = 0.7
p_validation = 0.15
p_test = 0.15
input_dims = [80,145]
output_dims = 7

"""
Model Hyper Parameters
    (optimizer is initialized in each run)
"""
epochs = 50
mini_batch_size = 90
loss = "binary_crossentropy"
dropout_p = 0.1
regularization_rate = 0.0001

archive_dir = "../data/live_samples.h5"
"""
Get our dataset object for easy reference of data subsets (training, validation, and test) from our archive_dir.
"""
dataset, whole_normalization_data = load_dataset_obj(p_training, p_validation, p_test, archive_dir, output_dims, whole_data_normalization=False)

"""
Get properly formatted input dimensions for our convolutional layer, so that we go from [h, w] to [-1, h, w, 1]
"""
image_input_dims = [-1, input_dims[0], input_dims[1], 1]

"""
Reshape our dataset inputs accordingly
"""
dataset.training.x = np.reshape(dataset.training.x, image_input_dims)
dataset.validation.x = np.reshape(dataset.validation.x, image_input_dims)
dataset.test.x = np.reshape(dataset.test.x, image_input_dims)

dataset.training.x = np.repeat(dataset.training.x, [3], axis=3)
dataset.validation.x = np.repeat(dataset.validation.x, [3], axis=3)
dataset.test.x = np.repeat(dataset.test.x, [3], axis=3)
print dataset.training.x.shape
print dataset.validation.x.shape
print dataset.test.x.shape

model = applications.VGG19(weights='imagenet', include_top=False)

bottleneck_training_features = model.predict(dataset.training.x)
bottleneck_validation_features = model.predict(dataset.validation.x)
bottleneck_test_features = model.predict(dataset.test.x)

with h5py.File("../saved_networks/bottleneck_features.h5", "w") as hf:
    hf.create_dataset("training", data=bottleneck_training_features)
    hf.create_dataset("validation", data=bottleneck_validation_features)
    hf.create_dataset("test", data=bottleneck_test_features)
