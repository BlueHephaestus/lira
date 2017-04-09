import os
import pickle
import numpy as np
import keras
from keras.models import load_model

class StaticConfig(object):
    def __init__(self, model, model_dir):
        """
        Arguments:
            model: filename of our model to load (no .h5 included)
            model_dir: filepath of our model

        Returns:
            Initialises class variables mean, std, and model.
            We load the metadata for mean and std,
            Then initialise the model using our Keras method.
        """

        """
        Get our absolute filepaths for model and metadata from our filename and source filepath
        """
        metadata_dir = "%s%s%s_metadata.pkl" % (model_dir, os.sep, model)
        model_dir = "%s%s%s.h5" % (model_dir, os.sep, model)

        """
        Load metadata from .pkl file
        """
        with open(metadata_dir, 'rb') as f:
            metadata = pickle.load(f)

        """
        Assign class attributes to necessary metadata
        """
        self.mean = metadata[0]
        self.std = metadata[1]
        
        """
        Finally, load our Keras model.
        """
        self.model = load_model(model_dir)

    def classify(self, x):
        """
        Arguments:
            x: a np array of the shape that the given model expects as input.

        Returns:
            Gets one-hot predictions from the input given, of shape (n, class_n)
            Then argmaxes, and returns a vector of shape (n,)
        """
        """
        Get predictions on batch with our given mini batch x
        """
        predictions = self.model.predict_on_batch(x)

        """
        Get the argmax of these predictions, so as to get a classification instead of a one-hot vector
        """
        predictions = np.argmax(predictions, axis=1)

        """
        Return these.
        """
        return predictions

