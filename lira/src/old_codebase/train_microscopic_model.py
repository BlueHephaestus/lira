"""
LIRA MK 2

This is the file for training our Microscopic Classification models. 

During development, we made use of many other metrics for debugging 
    and testing models. However now that we are done with development
    and this is the final agreed-upon model for Microscopic Classification,
    we no longer have these. They can however be found in the github commit history.

This file simply creates our model and trains it, using parameters
    as specified in the below documentation.

-Blake Edwards / Dark Element
"""
import os 

import h5py
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
from keras.regularizers import l2

def read_new(archive_dir):
    """
    Create necessary x.dat file so we don't crash, because I can not figure out the necessary write option in numpy to ensure that it makes the file for us.
    """
    open("x.dat","w").close()
    with h5py.File(archive_dir, "r", chunks=True, compression="gzip") as hf:
        """
        Load our X data the usual way,
            using a memmap for our x data because it may be too large to hold in RAM,
            and loading Y as normal since this is far less likely 
                -using a memmap for Y when it is very unnecessary would likely impact performance significantly.
        """
        #x_shape = tuple(hf.get("x_shape"))
        x_shape = (hf.get("x_shape")[0], 80, 145, 3)
        x = np.memmap("x.dat", dtype="float32", mode="r+", shape=x_shape)
        memmap_step = 1000
        hf_x = hf.get("x")
        for i in range(0, x_shape[0], memmap_step):
            x[i:i+memmap_step] = np.reshape(hf_x[i:i+memmap_step], (-1, 80, 145, 3))
            print i
        y = np.array(hf.get("y")).astype(int)
    return x, y

def to_categorical(vec, width):
    """
    Arguments:
        vec: Vector of indices
        width: Width of our categorical matrix, should be the max value in our vector (but could also be larger if you want).

    Returns:
        A one hot / categorical matrix, so that each entry is the one-hot vector of the index, e.g.
            2 (with width = 4) -> [0, 0, 1, 0]
    """
    categorical_mat = np.zeros((len(vec), width))
    categorical_mat[np.arange(len(vec)), vec] = 1
    return categorical_mat

def train_model(model_title, model_dir="../saved_networks", archive_dir="../data/live_samples.h5"):
    """
    Arguments:
        model_title: String to be converted to file name for saving our model
        model_dir: Filepath to store our model
        archive_dir: Filepath and filename for h5 file to load our samples from, for training

    Returns:
        Saves our Microscopic model as a Keras model trained on this data to the model filepath.
    """

    """
    Load our X and Y data
    """
    x, y = read_new(archive_dir)
    print "Input Data Shape:", x.shape

    """
    Data Parameters
    """
    input_dims = [80, 145, 3]
    output_dims = class_n = 4

    y = to_categorical(y, class_n)

    """
    Model Hyper Parameters
    """
    loss = "binary_crossentropy"
    optimizer = Adam(1e-4)
    regularization_rate = 1e-5
    epochs = 50
    mini_batch_size = 100

    """
    Output Parameters
    """
    output_title = model_title
    output_dir = model_dir
    output_filename = output_title.lower().replace(" ", "_")

    """
    Get properly formatted input dimensions for our convolutional layer, so that we go from [h, w] to [-1, h, w, 1]
    """
    image_input_dims = [-1]
    image_input_dims.extend(input_dims)

    x = np.reshape(x, image_input_dims)

    """
    Define our model
    """
    model = Sequential()

    #input 80,145,3
    model.add(Conv2D(20, (7, 12), padding="valid", input_shape=input_dims, data_format="channels_last", activation="sigmoid", kernel_regularizer=l2(regularization_rate)))
    #input 74, 134, 20
    model.add(MaxPooling2D(data_format="channels_last"))

    #input 37,67,20
    model.add(Conv2D(40, (6, 10), padding="valid", data_format="channels_last", activation="sigmoid", kernel_regularizer=l2(regularization_rate)))
    #input 32,58, 40
    model.add(MaxPooling2D(data_format="channels_last"))

    #input 16,29,40
    model.add(Flatten())
    
    #input 18560
    model.add(Dense(1024, activation="sigmoid", kernel_regularizer=l2(regularization_rate)))
    #input 1024
    model.add(Dense(100, activation="sigmoid", kernel_regularizer=l2(regularization_rate)))
    #input 100
    model.add(Dense(output_dims, activation="softmax", kernel_regularizer=l2(regularization_rate)))

    """
    Compile our model with our previously defined loss and optimizer, and recording the accuracy on the training data.
    """
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

    """
    Use this to check your layer's input and output shapes, for checking your math / calculations / designs
    """
    print "Layer Input -> Output Shapes:"
    for layer in model.layers:
        print layer.input_shape, "->", layer.output_shape

    """
    Finally, train our model
    """
    outputs = model.fit(x, y, epochs=epochs, batch_size=mini_batch_size)

    """
    Save our model to an h5 file with our output filename
    """
    print "Saving Model..."
    model.save("%s%s%s.h5" % (model_dir, os.sep, output_filename))
