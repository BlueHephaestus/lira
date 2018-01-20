import os
import keras
from keras.models import load_model

class StaticConfig(object):
    def __init__(self, model, model_dir):
        """
        Arguments:
            model: filename of our model to load (no .h5 included)
            model_dir: filepath of our model

        Returns:
            We initialise the model using our Keras method, as a class variable.
        """

        """
        Get our absolute filepaths for model from our filename and source filepath
        """
        model_dir = "%s%s%s.h5" % (model_dir, os.sep, model)

        """
        Load our Keras model.
        """
        self.model = load_model(model_dir)

    def classify(self, x):
        """
        Arguments:
            x: a np array of the shape that the given model expects as input.

        Returns:
            Gets predictions from the input given, of shape (n, class_n),
                and returns this matrix.
        """
        """
        Get predictions on batch with our given mini batch x
        """
        predictions = self.model.predict_on_batch(x)

        """
        Return these.
        """
        return predictions

