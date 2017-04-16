import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.regularizers import l2
from keras.optimizers import Adam
from keras import applications

import dataset_handler
from dataset_handler import *

import keras_test_callback
from keras_test_callback import TestCallback
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
mini_batch_size = 100
loss = "binary_crossentropy"
dropout_p = 0.5
regularization_rate = 0.0001

archive_dir = "../data/live_samples.h5"
dataset, whole_normalization_data = load_dataset_obj(p_training, p_validation, p_test, archive_dir, output_dims, whole_data_normalization=False)

image_input_dims = [-1, input_dims[0], input_dims[1], 1]
with h5py.File("../saved_networks/bottleneck_features.h5", "r") as hf:
    dataset.training.x = np.array(hf.get("training"))
    dataset.validation.x = np.array(hf.get("validation"))
    dataset.test.x = np.array(hf.get("test"))

dataset.training.x = np.reshape(dataset.training.x, [-1, 64, 64])
dataset.validation.x = np.reshape(dataset.validation.x, [-1, 64, 64])
dataset.test.x = np.reshape(dataset.test.x, [-1, 64, 64])

model = Sequential()
model.add(Flatten(input_shape=[64, 64]))
model.add(Dense(2048, activation="relu", kernel_regularizer=l2(regularization_rate)))
model.add(Dropout(dropout_p))
model.add(Dense(1024, activation="relu", kernel_regularizer=l2(regularization_rate)))
model.add(Dropout(dropout_p))
model.add(Dense(512, activation="relu", kernel_regularizer=l2(regularization_rate)))
model.add(Dropout(dropout_p))
model.add(Dense(128, activation="relu", kernel_regularizer=l2(regularization_rate)))
model.add(Dropout(dropout_p))
model.add(Dense(output_dims, activation="softmax"))

model.compile(loss=loss, optimizer='rmsprop', metrics=["accuracy"])

test_callback = TestCallback(model, (dataset.test.x, dataset.test.y))

outputs = model.fit(dataset.training.x, dataset.training.y, validation_data=(dataset.validation.x, dataset.validation.y), callbacks=[test_callback], epochs=epochs, batch_size=mini_batch_size)

