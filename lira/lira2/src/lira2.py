"""
LIRA MK 2

Instead of continuing to hook into DENNIS MK6, another of my repositories,
    I realized that it was better to write a small amount of code for each problem rather than a massive amount of code for multiple problems.
    This way, anyone can look at these files and understand it much easier, rather than looking at 50 files to understand one problem.

This code still uses many of the ideas I had in DENNIS MK6 however, and has plenty of nice features for handling the training.
After building the model, it trains a changable amount of times to reduce variance between runs, 
    then saves the results,
    saves the model,
    saves the model metadata,
    and graphs the results (if enabled).

-Blake Edwards / Dark Element
"""
import pickle, os

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2

import dataset_handler
from dataset_handler import *

import results_grapher
from results_grapher import graph_results

import keras_test_callback
from keras_test_callback import TestCallback

def train_model(model_title, model_dir="../saved_networks", archive_dir="../data/live_samples.h5"):
    """
    Arguments:
        model_title: String to be converted to file name for saving our model
        model_dir: Filepath to store our model
        archive_dir: Filepath and filename for h5 file to load our samples from, for training

    Returns:
        After setting up and training our model on `run_count` independent training iterations, it then
            saves the results,
            saves the model,
            saves the model metadata,
            and graphs the results (if enabled).
    """
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
    """
    epochs = 100
    mini_batch_size = 96
    loss = "binary_crossentropy"
    optimizer = Adam(1e-4)
    dropout_p = 0.5
    regularization_rate = 0.000016

    """
    Amount of times we train our model to remove possible variance in the results
    """
    run_count = 1

    """
    Output Parameters
    """
    output_title = model_title
    output_dir = model_dir
    output_filename = output_title.lower().replace(" ", "_")
    graph_output = True

    """
    We train our model independently `run_count` times, 
        so as to remove as much variance as possible between iterations,
        and get closer to the true accuracy of our model.
        We have to include a lot in this loop unfortunately, since keras will start training on the model where it left off from the last run if not.

    Since each run we train for `epochs` epochs, and each epoch produces 
        training data loss, training data accuracy, validation data accuracy, and test data accuracy,

    We can initialize our results as a np zeroed array of size (run_count, epochs, 4),
        and set our results into this each run.
    """
    results = np.zeros((run_count, epochs, 4))
    for run_i in range(run_count):
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

        """
        Define our model
        """
        model = Sequential()
        model.add(Convolution2D(20, 7, 12, border_mode="valid", input_shape=(80, 145, 1), W_regularizer=l2(regularization_rate)))
        model.add(Activation("sigmoid"))
        model.add(MaxPooling2D())

        model.add(Convolution2D(40, 6, 10, border_mode="valid", W_regularizer=l2(regularization_rate)))
        model.add(Activation("sigmoid"))
        model.add(MaxPooling2D())

        model.add(Flatten())

        model.add(Dense(1024, W_regularizer=l2(regularization_rate)))
        model.add(Activation("sigmoid"))
        model.add(Dropout(dropout_p))

        model.add(Dense(100, W_regularizer=l2(regularization_rate)))
        model.add(Activation("sigmoid"))
        model.add(Dropout(dropout_p))

        model.add(Dense(output_dims, W_regularizer=l2(regularization_rate)))
        model.add(Activation("softmax"))

        """
        Compile our model with our previously defined loss and optimizer, and recording the accuracy on the training data.
        """
        model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

        """
        Get our test data callback with our previously imported class from keras_test_callback.py
            We also reset it every loop so that keras doesn't automatically append results to it
        """
        test_callback = TestCallback(model, (dataset.test.x, dataset.test.y))

        """
        Get our outputs by training on training data and evaluating on validation and test accuracy each epoch,
            as well as with our previously defined hyper-parameters
        """
        outputs = model.fit(dataset.training.x, dataset.training.y, validation_data=(dataset.validation.x, dataset.validation.y), callbacks=[test_callback], nb_epoch=epochs, batch_size=mini_batch_size)

        """
        Stack and transpose our results to get a matrix of size epochs x 4, where each row contains the statistics for that epoch.
        """
        results[run_i] = np.vstack((outputs.history["loss"], outputs.history["acc"], outputs.history["val_acc"], test_callback.acc)).transpose()

    """
    With our results now obtained for all runs and epochs, 
        We then save the results to a pkl file of format "`output_filename`_results.pkl"
    """
    print "Saving Results..."
    results_filename = "%s%s%s_results.pkl" % (model_dir, os.sep, output_filename)
    with open(results_filename, "wb") as f:
        pickle.dump((results), f, protocol=-1)

    """
    Save our model to an h5 file with our output filename
    """
    print "Saving Model..."
    model.save("%s%s%s.h5" % (model_dir, os.sep, output_filename))

    """
    Save our extra model metadata:
    For this example, our metadata solely consists of our whole normalization data, 
        assign it and save it to a pkl file of format "`output_filename`_metadata.pkl"
    """
    print "Saving Model Metadata..."
    metadata = whole_normalization_data
    metadata_filename = "%s%s%s_metadata.pkl" % (model_dir, os.sep, output_filename)
    with open(metadata_filename, "wb") as f:
        pickle.dump((metadata), f, protocol=-1)

    """
    We then graph our results:
    Since we might be running this without a graphical interface, which would cause an error,
        we surround this in a try-except so we don't break execution.
    """
    if graph_output:
        try:
            graph_results(results)
        except:
            pass
