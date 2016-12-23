import os, sys
import pickle, gzip
import numpy as np
#import tensorflow as tf
import keras
from keras.models import load_model

class StaticConfig(object):
    def __init__(self, nn, archive_dir):

        #Get directories
        nn_dir = "%s/../saved_networks/%s.h5" % (archive_dir, nn)
        metadata_dir = "%s/../saved_networks/%s_metadata.pkl.gz" % (archive_dir, nn)

        f = gzip.open(metadata_dir, 'rb')
        metadata = pickle.load(f)
        f.close()

        #Assign metadata values
        self.mean = metadata[0]
        self.stddev = metadata[1]
        model_layers = metadata[2]
        input_dims = metadata[3]
        output_dims = metadata[4]
        
        self.model = load_model(nn_dir)

    def classify(self, x):
        #predictions = np.argmax(self.model.predict_on_batch(x), axis=1)
        """
        Added to have a different color when we are less than 50% confident in our results
        """
        raw_predictions = self.model.predict_on_batch(x)
        predictions = np.zeros(shape=raw_predictions.shape[0])

        for raw_prediction_i, raw_prediction in enumerate(raw_predictions):
            predictions[raw_prediction_i] = np.argmax(raw_prediction)

        return predictions

