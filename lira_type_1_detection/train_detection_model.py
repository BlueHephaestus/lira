import h5py
import cv2
import numpy as np
import random

import keras
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.regularizers import l2

def recursive_get_type(x):
    try:
        return recursive_get_type(x[0])
    except:
        return type(x)

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

def create_transformation_matrix(theta, sx, sy, dx, dy, top_left_to_center, center_to_top_left):
    """
    Given human readable parameters for translation (dx & dy), scaling (sx & sy), and rotation (theta),
        We move the center of the image to the origin for more intuitive transformations,
        Then apply them in the order rotation -> translation -> scaling
    Note: theta given in radians
    """

    """
    Create our transformation matrices

    Rotation (no rotation: theta = 0):
        [cos(theta), -sin(theta), 0]
        [sin(theta), cos(theta),  0]
        [0         , 0         ,  1]

    Scaling (no scaling: sx = sx = 1):
        [sx        , 0         ,  0]
        [0         , sy        ,  0]
        [0         , 0         ,  1]

    Translation (no translating: dx = dy = 0):
        [1         , 0         , dx]
        [0         , 1         , dy]
        [0         , 0         ,  1]

    """
    rotation = np.float32([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0,0,1]])
    translation = np.float32([[1,0,dx],[0,1,dy],[0,0,1]])
    scaling = np.float32([[sx,0,0],[0,sy,0],[0,0,1]])

    """
    Create our final transformation matrix by chaining these together in a dot product of the form:
        res = top_left_to_center * scaling * translation * rotation * center_to_top_left.
    This is because we want to apply them in our order, rotation -> translation -> scaling. 
    However if we dot'd them normally, this would happen in the reverse order.
    So, to counter this, we reverse our order of dot product to be the above.
    """
    T = top_left_to_center.dot(scaling).dot(translation).dot(rotation).dot(center_to_top_left)
    return T

def get_new_balanced_batch(x, y, sample_n, positive_n, batch_size):

    while True:
        x_batch_shape = [batch_size]
        x_batch_shape.extend(x.shape[1:])
        x_batch = np.zeros(x_batch_shape)
        y_batch = np.zeros((batch_size, 2))

        """
        First get positive examples by picking batch_size/2 from our first positive_n samples.
        """
        positive_indices = random.sample(range(positive_n), batch_size/2)
        for i, positive_i in enumerate(positive_indices):
            x_batch[i] = x[positive_i]
            y_batch[i] = y[positive_i]

        """
        Then randomly choose batch_size/2 number of indices after our positive examples, 
            so we get an equal number of random samples from our negatives.
        We use positive_n so we don't accidentally choose positive samples where we need negative ones
        We use random.sample for this.
        """
        negative_indices = random.sample(range(positive_n, sample_n), batch_size/2)
        for i, negative_i in enumerate(negative_indices):
            x_batch[(batch_size/2)+i] = x[negative_i]
            y_batch[(batch_size/2)+i] = y[negative_i]

        """
        We now have a balanced amount of positives and negatives in our batch, 
        But we also want to randomly and independently augment each of them. 
        We do this so that our network learns to generalize better (this is how many state-of-the-art ImageNet models are trained),
            by never seeing the same example twice no matter how long it's trained.
        So, we want to randomly and independently augment each of them. 
        For ease, i'm going to write n = size of our batch.
        """
        """
        In order to augment n samples independently, we need n transformation matrices.
        Fortunately, we don't need to randomly generate the entire matrices, just 5 parameters for each:
            1) Translation X
            2) Translation Y
            3) Rotation
            4) Scale X
            5) Scale Y
        Since these are the same transformations we used when augmenting our data originally. 
        Once we have these 5 parameters, we could create 3 transformation matrices
            (one for translation, one for rotation, one for scale) and dot them together
            to get one resulting transformation matrix, 
            however since they are always of the same form and we want to minimize computation time here,
            we will manually combine the parameters to immediately create our resulting matrix.
        We also have to take precaution here since we are combining transformations - 
            we want to avoid removing the original data from the image entirely.
        In order to avoid this / minimize the chance of it happening, we have to use smaller
            ranges for each of the 5 transformation parameters:

            1) Translation X - [-16, 16] #Old Range - [-32, 32]
            2) Translation Y - [-16, 16] #Old Range - [-32, 32]
            3) Rotation - [0, 360] or [0, 2*pi] #This one's the same because rotation doesn't remove near as much data as the other two
            4) Scale X - [.84, 1.19] #Old Range - [.7, 1.4]
            5) Scale Y - [.84, 1.19] #Old Range - [.7, 1.4]

        We will generate all the parameters at once, then iteratively create the matrices for each.
        I debated creating all the transformation matrices at once, however I think this is very unintuitive for the small gain it might provide.
        """
        #We use uniform because it's simpler, .rand could also be used
        translations = np.random.uniform(-16, 16, size=(batch_size, 2))
        rotations = np.random.uniform(0, 2*np.pi, size=(batch_size, 1))
        scales = np.random.uniform(.84, 1.19, size=(batch_size, 2))

        #We also need our normalization matrices
        """
        Moves top left corner of frame to center of frame
        """
        h = x.shape[1]
        w = x.shape[2]
        top_left_to_center = np.array([[1., 0., .5*w], [0., 1., .5*h], [0.,0.,1.]])

        """
        Moves center of frame to top left corner of frame, the inverse of our previous transformation
        """
        center_to_top_left = np.array([[1., 0., -.5*w], [0., 1., -.5*h], [0.,0.,1.]])

        for i, sample in enumerate(x_batch):
            T = create_transformation_matrix(rotations[i], scales[i,0], scales[i,1], translations[i,0], translations[i,1], top_left_to_center, center_to_top_left)
            x_batch[i] = cv2.warpAffine(sample, T[:2], sample.shape[:2], borderValue=[244,244,244])

        yield x_batch, y_batch

def read_new(archive_dir):
    with h5py.File(archive_dir, "r", chunks=True, compression="gzip") as hf:
        """
        Load our X data the usual way,
            using a memmap for our x data because it may be too large to hold in RAM,
            and loading Y as normal since this is far less likely 
                -using a memmap for Y when it is very unnecessary would likely impact performance significantly.
        """
        #x_shape = tuple(hf.get("x_shape"))
        x_shape = (hf.get("x_shape")[0], 128, 128, 3)
        x = np.memmap("x.dat", dtype="float32", mode="r+", shape=x_shape)
        memmap_step = 1000
        hf_x = hf.get("x")
        for i in range(0, x_shape[0], memmap_step):
            x[i:i+memmap_step] = np.reshape(hf_x[i:i+memmap_step], (-1, 128, 128, 3))
            print i
        y = np.array(hf.get("y")).astype(int)
    return x, y

<<<<<<< HEAD
def train_model(archive_dir="augmented_samples.h5", model_dir="type1_detection_model_mk5"):
=======
def train_model(archive_dir, model_dir):
>>>>>>> 0b4dd0478ac0b5eecfde8789898381e427581835
    """
    Arguments:
        archive_dir: An h5py archive directory with an x, x_shape, y, and y_shape dataset; where
            x: training data np array of shape (n, f) where n is the number of samples and f is the number of features.
            x_shape: shape of the x array
            y: training data np array of shape (n,) where n is the number of samples
            y_shape: shape of the y array
        model_dir: Filepath to store our model

    Returns:
        Saves a Keras model trained on this data to the model filepath, 
    """
    """
    Load our X and Y data,
        using a memmap for our x data because it may be too large to hold in RAM,
        and loading Y as normal since this is far less likely 
            also, using a memmap for Y when it is very unnecessary would likely impact performance significantly.

    I noticed that assigning the memmap all at once would often still be too large to hold in RAM,
        so we step through the archive and assign it in sections at a time.
    We step through and assign sections based on memmap_step.
    """
    x, y = read_new(archive_dir)
    print "Input Data Shape:", x.shape
    #input_shape = [x.shape[1]]
    input_shape = [128,128,3]
    """
    Assumes our x array has the positive_n samples first, unshuffled,
    and the remaining samples are the negative samples.
    """
    sample_n = len(x)
    positive_n = np.sum(y==1)

    y = to_categorical(y, 2)

    loss = "binary_crossentropy"
    optimizer = Adam()
    regularization_rate = 1e-4
    epochs = 1000
    batch_size = 100

    model = Sequential()
    
    #input 128,128,3
    model.add(Conv2D(32, (7, 7), strides=(2, 2), padding="same", input_shape=input_shape, data_format="channels_last", activation="relu", kernel_regularizer=l2(regularization_rate)))
    #input 64,64,32
    model.add(MaxPooling2D(data_format="channels_last"))

    #input 32,32,32
    model.add(Conv2D(32, (3, 3), strides=(1, 1), padding="same", data_format="channels_last", activation="relu", kernel_regularizer=l2(regularization_rate)))

    #input 32,32,32
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same", data_format="channels_last", activation="relu", kernel_regularizer=l2(regularization_rate)))
    #input 32,32,64
    model.add(MaxPooling2D(data_format="channels_last"))

    #input 16,16,64
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same", data_format="channels_last", activation="relu", kernel_regularizer=l2(regularization_rate)))

    #input 16,16,64
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same", data_format="channels_last", activation="relu", kernel_regularizer=l2(regularization_rate)))

    #input 16,16,64
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same", data_format="channels_last", activation="relu", kernel_regularizer=l2(regularization_rate)))

    #input 16,16,64
    model.add(MaxPooling2D(data_format="channels_last"))

    #input 8,8,64
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same", data_format="channels_last", activation="relu", kernel_regularizer=l2(regularization_rate)))

    #input 8,8,64
    model.add(AveragePooling2D(pool_size=(8,8), data_format="channels_last"))

    #input 1,1,64
    model.add(Flatten())

    #input 1*1*64 = 64
    model.add(Dense(2, activation="softmax", kernel_regularizer=l2(regularization_rate)))
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

    #Use this to check your layer's input and output shapes, for checking your math / calculations / designs
    print "Layer Input -> Output Shapes:"
    for layer in model.layers:
        print layer.input_shape, "->", layer.output_shape
    model.fit_generator(get_new_balanced_batch(x, y, sample_n, positive_n, batch_size), steps_per_epoch=100, epochs=epochs)
    #model.fit_generator(get_new_balanced_batch(x, y, sample_n, positive_n), steps_per_epoch=100, epochs=epochs)

    predictions = model.predict(x)
    print np.sum(np.argmax(predictions, axis=1)==1)
    print np.sum(np.argmax(predictions, axis=1)==0)
    acc = np.sum(np.argmax(predictions, axis=1) == np.argmax(y, axis=1)) / float(sample_n)
    print "Raw Acc: ", acc
    """
    Save our model to the model filepath
    """
    print "Saving Model..."
    model.save("%s.h5" % (model_dir))

<<<<<<< HEAD
=======
train_model("augmented_samples.h5", "type1_detection_model_mk7")
>>>>>>> 0b4dd0478ac0b5eecfde8789898381e427581835
