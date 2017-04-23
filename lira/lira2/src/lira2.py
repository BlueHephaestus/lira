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
import numpy as np
np.random.seed(420)

import keras
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import *
from keras.models import Model

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
    input_dims = [80,145,3]
    output_dims = 7

    """
    Model Hyper Parameters
        (optimizer is initialized in each run)
    """
    epochs = 100
    mini_batch_size = 95
    loss = "binary_crossentropy"
    dropout_p = 0.7
    regularization_rate = 0.031623

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
        Since Keras was breaking on the second run when experimenting with lira2_bbho_configurer, it
            brought to my attention that Keras might be storing the past gradients within the Adam object,
            so when we tried to use the Adam object from our last run, it tries to use the data it stored there from the last run,
            and breaks. 
        
        So, while Keras doesn't break during the execution of this file, we initialize this here for each run to ensure test independence.
        """
        optimizer = Adam(1e-4)

        """
        Get our dataset object for easy reference of data subsets (training, validation, and test) from our archive_dir.
        """
        dataset, whole_normalization_data = load_dataset_obj(p_training, p_validation, p_test, archive_dir, output_dims, whole_data_normalization=False)

        """
        Get properly formatted input dimensions for our convolutional layer, so that we go from [h, w] to [-1, h, w, 1]
        """
        image_input_dims = [-1]
        image_input_dims.extend(input_dims)

        print dataset.training.x.shape
        """
        Reshape our dataset inputs accordingly
        """
        dataset.training.x = np.reshape(dataset.training.x, image_input_dims)
        dataset.validation.x = np.reshape(dataset.validation.x, image_input_dims)
        dataset.test.x = np.reshape(dataset.test.x, image_input_dims)

        """
        Since our last dimension is now 3 instead of 1, we update our image_input_dims
        """
        image_input_dims = [-1]
        image_input_dims.extend(input_dims)
        
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
        print "Training Bottleneck Model..."
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
        We then save the results to a pkl file of format "`output_filename`_results.pkl"
    """
    print "Saving Results..."
    results_filename = "%s%s%s_results.pkl" % (model_dir, os.sep, output_filename)
    with open(results_filename, "wb") as f:
        pickle.dump((results), f, protocol=-1)

    """
    Save our full model to an h5 file with our output filename
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
