"""
LIRA MK 2, pre-transfer-learning MK2

Instead of continuing to hook into DENNIS MK6, another of my repositories,
    I realized that it was better to write a small amount of code for each problem rather than a massive amount of code for multiple problems.
    This way, anyone can look at these files and understand it much easier, rather than looking at 50 files to understand one problem.

This code still uses many of the ideas I had in DENNIS MK6 however, and has plenty of nice features for handling the training.
After building the model, it trains a changable amount of times to reduce variance between runs, 
    then saves the results,
    saves the model,
    saves the model metadata,
    and graphs the results (if enabled).

This is also MK2 of the pre-transfer-learning LIRA MK 2.
In this one, we don't use validation or testing data,
    we don't use cross-validation to check results,
    and we don't graph the results, instead opting for
    using all of our data, a generator to ensure a balanced batch of data
    (where each batch is randomly augmented to ensure we never see the same sample twice and also helping to fight overfitting),
    and also an entirely new CNN model.
These upgrades are based on how I modeled the training for mk5 and mk6 of the object detector, also used with LIRA.

-Blake Edwards / Dark Element
"""
import pickle, os

import keras
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
from keras.regularizers import l2

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
        res = top_left_to_center * translation * scaling * shearing * rotation * center_to_top_left.
    This is because if we want to apply them in our order, we would normally get the inverse of the dot product
        (translation * scaling * shearing * rotation)^-1 ,
    But since inverses are computationally expensive we instead just flip the order, as that is equivalent in this scenario.
    We do not flip the order of the normalization transformations, however.
    """
    T = top_left_to_center.dot(scaling).dot(translation).dot(rotation).dot(center_to_top_left)
    return T

def get_new_multiclass_balanced_batch(x, y, sample_n, class_n, batch_size):

    while True:
        x_batch_shape = [batch_size]
        x_batch_shape.extend(x.shape[1:])
        x_batch = np.zeros(x_batch_shape)
        y_batch = np.zeros((batch_size, class_n))

        """
        Samples could be in any order, so instead of sorting them or putting them in individual arrays
            or keeping track of the index ranges or so on (which would be a pain in the ass the larger the class_n was),
            we're going to randomly choose from the array until we get a balanced batch.
        This should be just fine, since we have a relatively small batch size (usually only in the hundreds or less),
            and we can also get a list of all the indices in the array, shuffle them, and then loop through those.
        By doing the latter, we ensure we don't add duplicates and that our worst case is only n, not infinity.
        This strat ensures a very small memory overhead, and also has a very fast average case, both of which are super
            important, especially for this function.
        """
        """
        We could generate a random range and then shuffle it, or we can just use python's random.sample,
            which already returns a list of randomly selected, *unique* elements.
        So we can just tell it to select sample_n numbers from the range of sample_n. ezpz.
        """
        indices = random.sample(range(sample_n), sample_n)

        """
        We use this to know when we've got a balanced batch.
        We will be decrementing the numbers in each of them, until they're all 0. 
        """
        class_sample_counter = [int(np.floor(batch_size/float(class_n))) for i in range(class_n)]

        """
        Since the sum of these numbers may be less than the batch size however (at most class_n-1 less than the batch_size),
            we add 1 to each until the sum is equal to the batch size.
        This will only have to loop once because the sum of the numbers will always be 0 to class_n-1 less than the batch size.
        """
        i=0
        while np.sum(class_sample_counter) < batch_size:
            class_sample_counter[i] += 1
            i+=1

        """
        Now we can loop through our randomly generated indices. 
        """
        i=0
        for rand_i in indices:
            """
            For every one, we check if we need a sample of that class (with class_sample_counter), and if so,
                we put it in our batch and decrement the class_sample_counter for that class.
            """
            if class_sample_counter[y[rand_i]] > 0:
                x_batch[i] = x[rand_i]
                y_batch[i] = y[rand_i]
                class_sample_counter[y[rand_i]]-=1
                i+=1
        """
        So now we have an entire batch of samples, with the same amount of each class in that batch,
            plus or minus 1 due to integer division.
        We also want to randomly and independently augment each of them, however. 
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

            1) Translation X - [-10, 10] #Old Range - [-20, 20]
            2) Translation Y - [-18, 18] #Old Range - [-36, 36]
            3) Rotation - [0, 360] or [0, 2*pi] #This one's the same because rotation doesn't remove near as much data as the other two
            4) Scale X - [.84, 1.19] #Old Range - [.7, 1.4]
            5) Scale Y - [.84, 1.19] #Old Range - [.7, 1.4]

        We will generate all the parameters at once, then iteratively create the matrices for each.
        I debated creating all the transformation matrices at once, however I think this is very unintuitive for the small gain it might provide.
        """
        #We use uniform because it's simpler, .rand could also be used
        x_translations = np.random.uniform(-18, 18, size=(batch_size, 1))
        y_translations = np.random.uniform(-10, 10, size=(batch_size, 1))
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
            T = create_transformation_matrix(rotations[i], scales[i,0], scales[i,1], x_translations[i,0], y_translations[i,0], top_left_to_center, center_to_top_left)
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
        x_shape = (hf.get("x_shape")[0], 80, 145, 3)
        x = np.memmap("x.dat", dtype="float32", mode="r+", shape=x_shape)
        memmap_step = 1000
        hf_x = hf.get("x")
        for i in range(0, x_shape[0], memmap_step):
            x[i:i+memmap_step] = np.reshape(hf_x[i:i+memmap_step], (-1, 80, 145, 3))
            print i
            break
        y = np.array(hf.get("y")).astype(int)
    return x, y

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

    """
    Data Parameters
    """
    input_dims = [80, 145, 3]
    output_dims = class_n = 4

    y = to_categorical(y, class_n)

    """
    Model Hyper Parameters
        (optimizer is initialized in each run)
    """
    loss = "categorical_crossentropy"
    optimizer = Adam()
    regularization_rate = 1e-5
    epochs = 1000
    mini_batch_size = 1000

    """
    Output Parameters
    """
    output_title = model_title
    output_dir = model_dir
    output_filename = output_title.lower().replace(" ", "_")

    """
    Get properly formatted input dimensions for our convolutional layer, so that we go from [h, w] to [-1, h, w, 1]
    """
    image_input_dims = [-1]
    image_input_dims.extend(input_dims)

    x = np.reshape(x, image_input_dims)

    """
    Define our model
    """
    model = Sequential()

    #input 80,145,3
    model.add(Conv2D(32, (4, 7), strides=(2, 2), padding="same", input_shape=input_dims, data_format="channels_last", activation="relu", kernel_regularizer=l2(regularization_rate)))
    #input 40,73,32
    model.add(MaxPooling2D(data_format="channels_last"))

    #input 20,36,32
    model.add(Conv2D(32, (3, 3), strides=(1, 1), padding="same", data_format="channels_last", activation="relu", kernel_regularizer=l2(regularization_rate)))

    #input 20,36,32
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same", data_format="channels_last", activation="relu", kernel_regularizer=l2(regularization_rate)))
    #input 20,36,64
    model.add(MaxPooling2D(data_format="channels_last"))

    #input 10,18,64
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same", data_format="channels_last", activation="relu", kernel_regularizer=l2(regularization_rate)))

    #input 10,18,64
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding="same", data_format="channels_last", activation="relu", kernel_regularizer=l2(regularization_rate)))
    #input 10,18,128
    model.add(MaxPooling2D(data_format="channels_last"))

    #input 5,9,128
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding="same", data_format="channels_last", activation="relu", kernel_regularizer=l2(regularization_rate)))

    #input 5,9,128
    model.add(AveragePooling2D(pool_size=(5,9), data_format="channels_last"))

    #input 1,1,128
    model.add(Flatten())

    #input 1*1*128 = 128
    model.add(Dense(output_dims, activation="softmax", kernel_regularizer=l2(regularization_rate)))

    """
    Compile our model with our previously defined loss and optimizer, and recording the accuracy on the training data.
    """
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

    """
    Use this to check your layer's input and output shapes, for checking your math / calculations / designs
    """
    print "Layer Input -> Output Shapes:"
    for layer in model.layers:
        print layer.input_shape, "->", layer.output_shape

    model.fit_generator(get_new_multiclass_balanced_batch(x, y, sample_n, class_n, batch_size), steps_per_epoch=100, epochs=epochs)

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
