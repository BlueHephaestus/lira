"""
LIRA MK 2 for Configuration with BBHO - Bayesian Black-box Hyper-parameter Optimization
    Slightly modified from the original LIRA MK 2 file,
    in order to take hyper parameter arguments and not use some options, like graphing results or saving networks.

    Don't use this with the normal pipeline! It is only meant to be used with BBHO.

-Blake Edwards / Dark Element
"""
import pickle, os

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2

import dataset_handler
from dataset_handler import *

import results_grapher
from results_grapher import graph_results

import keras_test_callback
from keras_test_callback import TestCallback

import lira_optimization_input_handler
from lira_optimization_input_handler import handle_raw_hps

class Configurer(object):

    def __init__(self, epochs, run_count):
        """
        Arguments:
            epochs: Number of epochs to train our model in run_config
            run_count: Amount of times we train our model to remove possible variance in the results

        Returns:
            We initialize all of our HPs that are persistent across optimizations, 
                as well as accept some as configurable input that are persistent across independent training iterations. 
            Initialises these to class variables for our run_config method to use.
        """

        self.epochs = epochs
        self.run_count = run_count 

        """
        Data Parameters
            (optimizer is initialized in each run)
        """
        self.p_training = 0.7
        self.p_validation = 0.15
        self.p_test = 0.15
        self.input_dims = [80,145]
        self.output_dims = 7
        self.archive_dir=os.path.expanduser("~/programming/machine_learning/tuberculosis_project/lira/lira2/data/live_samples.h5")

    def run_config(self, hps):
        """
        Arguments:
            hps: A string of un-parsed hyper parameters, obtained from our BBHO optimizer. 
                These are problem specific, and are parsed (i.e. edge cases handled, converted to logarithmic scale, etc) in our lira_optimization_input_handler.py file.

        Returns:
            After setting up and training our model on `run_count` independent training iterations,
                using our hyper parameters, 
                we then return the average accuracy on the validation data over these training iterations. 

        We train our model independently `run_count` times, 
            so as to remove as much variance as possible between iterations,
            and get closer to the true accuracy of our model.
            We have to include a lot in this loop unfortunately, since keras will start training on the model where it left off from the last run if not.
        """

        """
        Our given hyper parameters are problem specific, and are parsed (i.e. edge cases handled, converted to logarithmic scale, etc) in our lira_optimization_input_handler.py file.
        """
        mini_batch_size, regularization_rate, dropout_p, activation_fn, loss, hp_str = handle_raw_hps(hps)

        """
        Since each run we train for `epochs` epochs, and each epoch produces 
            training data loss, training data accuracy, validation data accuracy, and test data accuracy,

        We can initialize our results as a np zeroed array of size (run_count, epochs, 4),
            and set our results into this each run.
        """
        results = np.zeros((self.run_count, self.epochs, 4))

        for run_i in range(self.run_count):
            """
            Keras breaks on the second run if we don't initialize our optimizer each time we run this.
                I'm not exactly sure why, but my theory is that it is storing the past gradients within the Adam object,
                so when we try to use the Adam object from our last run, it tries to use the data it stored there from the last run,
                and breaks. 
            
            So, we gotta initialize this for each run to both ensure test independence and to ensure keras doesn't die on us.
            """
            optimizer = Adam(1e-4)

            """
            Get our dataset object for easy reference of data subsets (training, validation, and test) from our archive_dir.
            """
            dataset, whole_normalization_data = load_dataset_obj(self.p_training, self.p_validation, self.p_test, self.archive_dir, self.output_dims, whole_data_normalization=False)

            """
            Get properly formatted input dimensions for our convolutional layer, so that we go from [h, w] to [-1, h, w, 1]
            """
            image_input_dims = [-1, self.input_dims[0], self.input_dims[1], 1]

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
            model.add(Conv2D(20, (7, 12), padding="valid", input_shape=(80, 145, 1), data_format="channels_last", activation=activation_fn, kernel_regularizer=l2(regularization_rate)))
            model.add(MaxPooling2D(data_format="channels_last"))

            model.add(Conv2D(40, (6, 10), padding="valid", data_format="channels_last", activation=activation_fn, kernel_regularizer=l2(regularization_rate)))
            model.add(MaxPooling2D(data_format="channels_last"))

            model.add(Flatten())

            model.add(Dense(1024, activation=activation_fn, kernel_regularizer=l2(regularization_rate)))
            model.add(Dropout(dropout_p))

            model.add(Dense(100, activation=activation_fn, kernel_regularizer=l2(regularization_rate)))
            model.add(Dropout(dropout_p))

            model.add(Dense(self.output_dims, activation=activation_fn, kernel_regularizer=l2(regularization_rate)))

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
            outputs = model.fit(dataset.training.x, dataset.training.y, validation_data=(dataset.validation.x, dataset.validation.y), callbacks=[test_callback], epochs=self.epochs, batch_size=mini_batch_size)

            """
            Stack and transpose our results to get a matrix of size epochs x 4, where each row contains the statistics for that epoch.
            """
            results[run_i] = np.vstack((outputs.history["loss"], outputs.history["acc"], outputs.history["val_acc"], test_callback.acc)).transpose()

        """
        With our results now obtained for all runs and epochs, 
            we then return the average validation accuracy, averaged across all runs.

        Since results is currently an np array of shape (run_count, epochs, 4), we can 
            parse out validation accuracy, then average it.
        """
        training_loss, training_acc, validation_acc, test_acc = np.split(results, 4, axis=2)

        """
        Then get the average over our runs
        """
        avg_validation_acc = np.mean(validation_acc, axis=0)
        
        """
        Then flatten it if needed, to remove any extraneous (1,) dimensions.
        """
        avg_validation_acc = validation_acc.flatten()
        return avg_validation_acc
