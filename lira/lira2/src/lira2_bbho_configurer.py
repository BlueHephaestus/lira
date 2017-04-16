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
        Static Hyperparameters
        """
        self.loss = "binary_crossentropy"

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
        mini_batch_size, regularization_rate, dropout_p, hp_str = handle_raw_hps(hps)

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
            Convert our data from grayscale into RGB by repeating the last dimension,
                for use with large pretrained networks, trained on rgb data.
            This goes from [-1, ..., 1] to [-1, ..., 3]
            """
            dataset.training.x = np.repeat(dataset.training.x, [3], axis=3)
            dataset.validation.x = np.repeat(dataset.validation.x, [3], axis=3)
            dataset.test.x = np.repeat(dataset.test.x, [3], axis=3)

            """
            Since our last dimension is now 3 instead of 1, we update our image_input_dims
            """
            image_input_dims = [-1, input_dims[0], input_dims[1], 3]
            
            """
            Open our pre-trained very deep network,
                without the dense layers at the end of the network,
                and with the input shape of our data
            """
            pretrained_model = VGG19(weights='imagenet', include_top=False, input_shape=image_input_dims[1:])

            """
            Get the features produced by our bottleneck layer, the features that are produced 
                by the last convolutional + maxpooling layer in this pretrained net (before the dense layers).
            These will be referred to as bottleneck features.
            """
            print "Generating Features from Pre-Trained Model..."
            dataset.training.x = pretrained_model.predict(dataset.training.x)
            dataset.validation.x = pretrained_model.predict(dataset.validation.x)
            dataset.test.x = pretrained_model.predict(dataset.test.x)

            """
            Now we can define a new, smaller model on top of this pretrained model.
            We will call this model the bottleneck model, for lack of a better name (who wants to call it "top_model"? that sucks)
                We set input shape to the output shape of our pretrained model, since it uses the bottleneck layer's output (the bottleneck features) as input.
            """
            bottleneck_model = Sequential()
            bottleneck_model.add(Flatten(input_shape=dataset.training.x.shape[1:]))
            bottleneck_model.add(Dense(1024, activation="relu", kernel_regularizer=l2(regularization_rate)))
            bottleneck_model.add(Dropout(dropout_p))
            bottleneck_model.add(Dense(128, activation="relu", kernel_regularizer=l2(regularization_rate)))
            bottleneck_model.add(Dropout(dropout_p))
            bottleneck_model.add(Dense(output_dims, activation="softmax"))

            """
            Compile our model with our previously defined loss and optimizer, and record the accuracy on the training data.
            """
            bottleneck_model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

            """
            Get our test data callback with our previously imported class from keras_test_callback.py
                We also reset it every loop so that keras doesn't automatically append results to it
            """
            test_callback = TestCallback(bottleneck_model, (dataset.test.x, dataset.test.y))

            """
            Get our outputs by training on training data and evaluating on validation and test accuracy each epoch,
                and use our previously defined hyper-parameters where needed.
            """
            outputs = bottleneck_model.fit(dataset.training.x, dataset.training.y, validation_data=(dataset.validation.x, dataset.validation.y), callbacks=[test_callback], epochs=epochs, batch_size=mini_batch_size)

            """
            Now that we have a trained bottleneck model on top of our pre-trained model,
                we add the bottleneck model to the literal top of our pre-trained model to get a new model for our problem.
            Our chain for the new, full, combined model is as follows:
                Inputs -> Pre-Trained Model -> Bottleneck Model -> Outputs
            So that:
                the input to the pretrained model is the input to the full model,
                the output of the pretrained model is the input to the bottleneck model,
                    (since it was trained on the bottleneck features / output of the pretrained model)
                and the output of the bottleneck model is the output of the full model.
            And we do this by symbolically linking the inputs and outputs according to our chain:
                Input -> pretrained outputs -> bottleneck outputs -> Output
            And can then initialize a full model using this linked input and output.

            If I didn't explain this well, please let me know.
            """
            pretrained_inputs = Input(image_input_dims[1:])
            pretrained_outputs = pretrained_model(pretrained_inputs)
            bottleneck_outputs = bottleneck_model(pretrained_outputs)
            model = Model(inputs=pretrained_inputs, outputs=bottleneck_outputs)
            """
            Note: Keras will give us a warning for not compiling our model, but this is fine because we aren't training the entire model.
            If you do wish to train the model, simply compile it with parameters/arguments of your choice.
            """

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
