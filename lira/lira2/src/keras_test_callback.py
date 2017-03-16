"""
Simple keras callback so that we can evaluate our model on the test data at the end of each epoch.

-Blake Edwards / Dark Element
"""
import keras
from keras.callbacks import Callback

class TestCallback(Callback):
    def __init__(self, model, test_data):
        self.model = model
        self.test_data = test_data
        self.loss = []
        self.acc = []

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        #print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
        self.loss.append(loss)
        self.acc.append(acc)
