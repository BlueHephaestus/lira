"""
Quick script to generate an augmented_samples.h5 file of augmented samples.
    Using live_samples.h5 and transformation_handler.py, 
        it creates an augmented_samples.h5 file by applying an arbitrary number of randomly generated
        affine transformation matrices, and also saves these matrices in a transformation_matrices.pkl file.

    Augmenting the original training data is a common practice used to help train neural networks that are
        more robust to changes in their input, so that they can learn to recognize the same sample if it's
        rotated, reflected, and so on.

-Blake Edwards / Dark Element
"""
import numpy as np
import cv2
import h5py, pickle

import transformation_handler
from transformation_handler import *

import dataset_handler
from dataset_handler import *

def undersample_dataset(x, y, class_n, undersample_n=None):
    """
    Arguments:
        x, y: Our dataset, x is input data and y is output data. Both are np arrays.
        class_n: Integer number of classes we have for our model.
        undersample_n: Integer number to sample our dataset down to, instead of the number of samples in the minority class. 
            If not supplied, will default to the number of samples in the minority class.

    Returns:
        Use this if your dataset is very biased, e.g. 95% one classification, 5% another classification.
        We first shuffle the dataset, 
        Then balance it by removing data from all classifications but the classification with the smallest number of labels,
        so that all classifications have the same number at the end.
    """
    unison_shuffle(x, y)

    """
    Array for our numbers of each class's labeled samples, we assign accordingly
    """
    class_sample_nums = np.zeros((class_n))
    for class_i in range(class_n):
        class_sample_nums[class_i] = np.sum(y==class_i)

    """
    We then get the minority / smallest class number, or use our given int if it's supplied
    """
    if undersample_n:
        class_sample_minority = undersample_n
    else:
        class_sample_minority = np.min(class_sample_nums)

    """
    Then we get a zeroed array to keep track of each class as we add it to our balanced dataset
    """
    class_sample_counter = np.zeros_like(class_sample_nums)

    """
    We find how large our array will be by using min(existing #, minority #),
        since if the number of samples for one class is > our minority #, it will be rounded down to our minority #,
        and if it is < our minority, it will be left as is.
    Using this, we can have class_sample_minority be an arbitrary value other than just the smallest value.
    """
    balanced_n = 0
    for class_sample_num in class_sample_nums:
        balanced_n += np.minimum(int(class_sample_num), int(class_sample_minority))

    """
    We then get our dimensions accordingly, and initialise each to zeros
    """
    balanced_x_dims = [balanced_n]
    balanced_x_dims.extend(x.shape[1:])
    """
    We then initialize our balanced arrays. Since these may be easily be far larger than our RAM,
        we use np.memmap, which is a way of holding them on disk and loading sections into RAM as we need them.
        They basically let us hold data structures in memory that are larger than our RAM, in case these are.
    They do need our dimensions as tuples, so we do a quick convert when creating that object.
    Otherwise, we create the balanced x and y arrays.
    """
    balanced_x = np.memmap("balanced_x.dat", dtype="float32", mode="w+", shape=tuple(balanced_x_dims))
    balanced_y = np.memmap("balanced_y.dat", dtype="float32", mode="w+", shape=(balanced_n,))

    """
    We then loop through, making sure to only add each class class_sample_minority times,
        and no more.
    """
    balanced_i = 0
    for sample_i, sample in enumerate(x):
        if class_sample_counter[y[sample_i]] < class_sample_minority:
            class_sample_counter[y[sample_i]]+=1
            balanced_x[balanced_i], balanced_y[balanced_i] = sample, y[sample_i]
            balanced_i += 1
    """
    We then return our new dataset
    """
    return balanced_x, balanced_y

def oversample_dataset(x, y, class_n):
    """
    Arguments:
        x, y: Our dataset, x is input data and y is output data. Both are np arrays.
        class_n: Integer number of classes we have for our model.

    Returns:
        Use this if your dataset is very biased, e.g. 95% one classification, 5% another classification.
        We balance the dataset by creating copy data for all classifications but the classification with the highest number of labeled samples,
        so that all classifications have the same number at the end and thus get represented equally.
    """

    """
    Array for our numbers of each class's labeled samples, we assign accordingly
    """
    class_sample_nums = np.zeros((class_n))
    for class_i in range(class_n):
        class_sample_nums[class_i] = np.sum(y==class_i)

    """
    We then get the majority / largest class number
    """
    class_sample_majority = np.max(class_sample_nums)

    """
    We then get our dimensions accordingly, and initialise each to zeros
    """
    balanced_x_dims = [int(class_sample_majority*class_n)]
    balanced_x_dims.extend(x.shape[1:])

    """
    We then initialize our balanced arrays. Since these may be easily be far larger than our RAM,
        we use np.memmap, which is a way of holding them on disk and loading sections into RAM as we need them.
        They basically let us hold data structures in memory that are larger than our RAM, in case these are.
    They do need our dimensions as tuples, so we do a quick convert when creating that object.
    Otherwise, we create the balanced x and y arrays.
    """
    balanced_x = np.memmap("balanced_x.dat", dtype="float32", mode="w+", shape=tuple(balanced_x_dims))
    balanced_y = np.memmap("balanced_y.dat", dtype="float32", mode="w+", shape=(int(class_sample_majority*class_n),))

    """
    Then we get a zeroed array to keep track of each class as we add it to our balanced dataset
    """
    class_sample_counter = np.zeros_like(class_sample_nums)

    """
    We then loop through and add each to our new array,
        repeatedly looping through our dataset until we have added enough to have the same number of samples for all classes.
    That is, until we don't correct any, at which point we know it is balanced.
    """
    balanced = False
    balanced_i = 0
    while not balanced:
        balanced = True
        for sample_i, sample in enumerate(x):
            if class_sample_counter[y[sample_i]] < class_sample_majority:
                class_sample_counter[y[sample_i]]+=1
                balanced_x[balanced_i], balanced_y[balanced_i] = sample, y[sample_i]
                balanced_i += 1
                balanced = False
    """
    We then return our new dataset
    """
    return balanced_x, balanced_y

def custom_sample_dataset(x, y, class_n, sample_ns):
    """
    Arguments:
        x, y: Our dataset, x is input data and y is output data. Both are np arrays.
        class_n: Integer number of classes we have for our model.
        sample_ns: List of integers containing the number of samples we want of each class when finished.
            e.g. if we had sample nums of [12, 6836] and put in [50, 60] for sample_ns, we'd end up with 50 of our first class and 60 of our second class.

    Returns:
        We balance the dataset by copying samples in classes that have less than their number specified by sample_ns, 
            and removing samples from classifications which have more than their number specified by sample_ns.
    """

    """
    Array for our numbers of each class's labeled samples, we assign accordingly
    """
    class_sample_nums = np.zeros((class_n))
    for class_i in range(class_n):
        class_sample_nums[class_i] = np.sum(y==class_i)

    """
    We then get our dimensions accordingly, and initialise each to zeros
    """
    """
    We then get our dimensions as the sum of our sample_ns
    """
    balanced_x_dims = [np.sum(sample_ns)]
    balanced_x_dims.extend(x.shape[1:])

    """
    We then initialize our balanced arrays. Since these may be easily be far larger than our RAM,
        we use np.memmap, which is a way of holding them on disk and loading sections into RAM as we need them.
        They basically let us hold data structures in memory that are larger than our RAM, in case these are.
    They do need our dimensions as tuples, so we do a quick convert when creating that object.
    Otherwise, we create the balanced x and y arrays.
    """
    balanced_x = np.memmap("balanced_x.dat", dtype="float32", mode="w+", shape=tuple(balanced_x_dims))
    balanced_y = np.memmap("balanced_y.dat", dtype="float32", mode="w+", shape=(np.sum(sample_ns),))

    """
    Then we get a zeroed array to keep track of each class as we add it to our balanced dataset
    """
    class_sample_counter = np.zeros_like(class_sample_nums)

    """
    We then loop through and add each to our new array,
        repeatedly looping through our dataset until we have added enough to have the same number of samples for all classes.
    That is, until we don't correct any, at which point we know it is balanced.
    """
    balanced = False
    balanced_i = 0
    while not balanced:
        balanced = True
        for sample_i, sample in enumerate(x):
            """
            Then, check if it is below our desired sample number for this class, and if so, oversample it.
            """
            if class_sample_counter[y[sample_i]] < sample_ns[y[sample_i]]:
                class_sample_counter[y[sample_i]]+=1
                balanced_x[balanced_i], balanced_y[balanced_i] = sample, y[sample_i]
                balanced_i += 1
                balanced = False
            """
            Then, check if it is above our desired sample number for this class, and if so, undersample it.
            """
            if class_sample_counter[y[sample_i]] > sample_ns[y[sample_i]]:
                class_sample_counter[y[sample_i]]-=1
                balanced_x[balanced_i], balanced_y[balanced_i] = sample, y[sample_i]
                balanced_i += 1
                balanced = False
                
    """
    We then return our new dataset
    """
    return balanced_x, balanced_y

def generate_augmented_data(archive_dir, augmented_archive_dir, metadata_dir, class_n, h=80, w=145, sigma=0.1, random_transformation_n=0, border_value=240, static_transformations=True, static_transformation_n=5, undersample_balance_dataset=False, undersample_n=None, oversample_balance_dataset=False, custom_sample_balance_dataset=False, sample_ns=[], rgb=False):
    """
    Arguments:
        archive_dir: String where .h5 file is stored containing model's data.
        augmented_archive_dir: String where .h5 file will be stored containing model's augmented data.
        metadata_dir: String where .pkl file will be stored containing transformation matrices after we augment our data
        class_n: Integer number of classes we have for our model.
        h, w: Height and width of each of our samples. Defaults to 80x145 for our LIRA project.
        sigma: Variance parameter to control how large our applied transformations are. Defaults to 0.1
        random_transformation_n: Number of random transformations to generate. Defaults to 0
        border_value: Value to pad missing parts of our image if we transform it off the viewport, 0-255
        static_transformations: Boolean to decide if we want to use our 5 preset transformations for augmentation or not. Defaults to True.
        static_transformation_n: Integer number of our static transformations to use, 0-5, defaults to 5.
        undersample_balance_dataset: Boolean to decide if we want to balance our dataset by removing enough of our majority samples to make all the classes have the same # of labeled classes. Defaults to False.
        oversample_balance_dataset: Boolean to decide if we want to balance our dataset by copying enough of our minority samples to make all the classes have the same # of labeled classes. Defaults to False.
        custom_sample_balance_dataset: Boolean to decide if we want to balance our dataset via a hard-coded custom technique. Defaults to False.
        rgb: Boolean for if we are handling rgb images (True), or grayscale images (False).

    Returns:
        After opening our samples from archive_dir, and initialising and normalising our static transformations if enabled,
            the methods in transformation_handler.py are used for generating and applying our transformations.
        We then store our transformation matrices (static and generated) into our metadata_dir, 
        And store our augmented dataset into our augmented_archive_dir.

    """

    with h5py.File(archive_dir, "r") as hf:
        """
        Load our X and Y data,
            using a memmap for our x data because it may be too large to hold in RAM,
            and loading Y as normal since this is far less likely 
                also, using a memmap for Y when it is very unnecessary would likely impact performance significantly.
        """

        x_shape = tuple(hf.get("x_shape"))
        x = np.memmap("x.dat", dtype="float32", mode="w+", shape=x_shape)
        x[:] = hf.get("x")[:]

        y_shape = tuple(hf.get("y_shape"))
        y = np.array(hf.get("y"))

    if undersample_balance_dataset:
        """
        Undersample our dataset if our option is enabled.
        """
        print "Balancing Dataset..."
        x, y = undersample_dataset(x, y, class_n, undersample_n)

    elif oversample_balance_dataset:
        """
        Oversample our dataset if our option is enabled.
        """
        print "Balancing Dataset..."

        x, y = oversample_dataset(x, y, class_n)
    elif custom_sample_balance_dataset:
        """
        Custom sample our dataset if our option is enabled.
        """
        print "Balancing Dataset..."
        x, y = custom_sample_dataset(x, y, class_n, sample_ns)

    """
    We get the h and w of each sample via the first, 
        and then we reshape so that we have images to properly transform.
    We first reshape our x by the h and w arguments passed in.
    """
    if rgb:
        x = np.reshape(x, (-1, h, w, 3))
    else:
        x = np.reshape(x, (-1, h, w))


    if static_transformations:
        """
        If enabled,
        Our preset static transformations are as follows:
        """
        transformation_matrices = np.array(
                [
                    [[0.866025403784,0.5,0.0],
                    [-0.5,0.866025403784,0.0],
                    [0.0,0.0,1.0]],

                    [[0.5,0.866025403784,0.0],
                    [-0.866025403784,0.5,0.0],
                    [0.0,0.0,1.0]],

                    [[6.12323399574e-17,1.0,0.0],
                    [-1.0,6.12323399574e-17,0.0],
                    [0.0,0.0,1.0]],

                    [[-0.5,0.866025403784,0.0],
                    [-0.866025403784,-0.5,0.0],
                    [0.0,0.0,1.0]],

                    [[-0.866025403784,0.5,0.0],
                    [-0.5,-0.866025403784,0.0],
                    [0.0,0.0,1.0]],

                    [[-1.0,1.22464679915e-16,0.0],
                    [-1.22464679915e-16,-1.0,0.0],
                    [0.0,0.0,1.0]],

                    [[-0.866025403784,-0.5,0.0],
                    [0.5,-0.866025403784,0.0],
                    [0.0,0.0,1.0]],

                    [[-0.5,-0.866025403784,0.0],
                    [0.866025403784,-0.5,0.0],
                    [0.0,0.0,1.0]],

                    [[-1.83697019872e-16,-1.0,0.0],
                    [1.0,-1.83697019872e-16,0.0],
                    [0.0,0.0,1.0]],

                    [[0.5,-0.866025403784,0.0],
                    [0.866025403784,0.5,0.0],
                    [0.0,0.0,1.0]],

                    [[0.866025403784,-0.5,0.0],
                    [0.5,0.866025403784,0.0],
                    [0.0,0.0,1.0]],

                    [[0.7, 0.0, 0.0],
                    [0.0, 0.7, 0.0],
                    [0.0, 0.0, 1.0]],

                    [[0.8, 0.0, 0.0],
                    [0.0, 0.8, 0.0],
                    [0.0, 0.0, 1.0]],

                    [[0.9, 0.0, 0.0],
                    [0.0, 0.9, 0.0],
                    [0.0, 0.0, 1.0]],

                    [[1.1, 0.0, 0.0],
                    [0.0, 1.1, 0.0],
                    [0.0, 0.0, 1.0]],

                    [[1.2, 0.0, 0.0],
                    [0.0, 1.2, 0.0],
                    [0.0, 0.0, 1.0]],

                    [[1.3, 0.0, 0.0],
                    [0.0, 1.3, 0.0],
                    [0.0, 0.0, 1.0]],

                    [[1.4, 0.0, 0.0],
                    [0.0, 1.4, 0.0],
                    [0.0, 0.0, 1.0]],

                    [[1.0, 0.0, 0.0],
                    [0.0, 1.0, 10.0],
                    [0.0, 0.0, 1.0]],

                    [[1.0, 0.0, 18.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]],

                    [[1.0, 0.0, 0.0],
                    [0.0, 1.0, -10.0],
                    [0.0, 0.0, 1.0]],

                    [[1.0, 0.0, -18.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]],

                    [[1.0, 0.0, 18.0],
                    [0.0, 1.0, 10.0],
                    [0.0, 0.0, 1.0]],

                    [[1.0, 0.0, 18.0],
                    [0.0, 1.0, -10.0],
                    [0.0, 0.0, 1.0]],

                    [[1.0, 0.0, -18.0],
                    [0.0, 1.0, 10.0],
                    [0.0, 0.0, 1.0]],

                    [[1.0, 0.0, -18.0],
                    [0.0, 1.0, -10.0],
                    [0.0, 0.0, 1.0]],
                ]
             )
        transformation_matrices = transformation_matrices[:static_transformation_n]

        """
        Initialize our normalization matrices.
            To illustrate why we do this, here is an example:
            For a 90-degree rotation:
                Originally:
                    This would normally be about the top-left corner of the image.
                    This is really not what we want, especially if we are combining transformations together.
                    We also by default lose the entire image, as it disappears from frame.
                    So, instead, we want to rotate around the center of the image.
                With Normalization:
                    By moving the image center to the top-left corner as the first transformation, 
                    we can then rotate in this way, so that we can easily apply transformations the 
                    human-understandable and human-manipulatable way.
                    So, we move top left corner so that it is the center, 
                        apply the transformation(s), 
                        and then move the center of the frame so that it is the center again.
                    We do this by dot'ing each of our transformation_matrices with these, 
                        since the composition of any affine transformations can be represented by the dot product
                        of the affine transformation matrices.
        """
        """
        Moves top left corner of frame to center of frame
        """
        top_left_to_center = np.array([[1., 0., .5*w], [0., 1., .5*h], [0.,0.,1.]])

        """
        Moves center of frame to top left corner of frame, the inverse of our previous transformation
        """
        center_to_top_left = np.array([[1., 0., -.5*w], [0., 1., -.5*h], [0.,0.,1.]])

        """
        Then loop through and apply our normalizations
        """
        for transformation_matrix_i, transformation_matrix in enumerate(transformation_matrices):
            transformation_matrices[transformation_matrix_i] = top_left_to_center.dot(transformation_matrix).dot(center_to_top_left)
    else:
        """
        If not enabled, we set our transformation_matrices so that they will be ignored in our augmenting functions in transformation_handler
        """
        transformation_matrices = []

    """
    Use the functions in our transformation_handler to handle our x and y parts of our dataset.
    """
    print "Augmenting Dataset..."
    x, transformation_matrices = generate_2d_transformed_data(x, sigma, random_transformation_n, transformation_matrices, border_value)
    y = generate_transformed_references(y, len(transformation_matrices))

    """
    We then reshape to our flattened dimension as we had originally
    """
    if rgb:
        x = np.reshape(x, (-1, h*w, 3))
    else:
        x = np.reshape(x, (-1, h*w))


    print "Writing Balanced / Augmented Dataset..."
    """
    We are now done augmenting our dataset,
        We now save our transformation matrices to the metadata_dir,
    """
    with open(metadata_dir, "w") as f:
        pickle.dump(transformation_matrices, f)

    """
    And finally create our augmented archive and store our transformed x and transformed y there.
    """
    with h5py.File(augmented_archive_dir, "w", chunks=True, compression="gzip") as hf:
        hf.create_dataset("x", data=x)
        hf.create_dataset("x_shape", data=x.shape)
        hf.create_dataset("y", data=y)
        hf.create_dataset("y_shape", data=y.shape)

#generate_augmented_data("../data/rgb_rim_samples.h5", "../data/model_2_samples.h5", "../data/transformation_matrices.pkl", 7, sigma=0.1, random_transformation_n=0, static_transformations=False, static_transformation_n=0, custom_sample_balance_dataset=True, sample_ns = [12159, 0, 4068, 7000, 17680, 0, 0], rgb=True)
generate_augmented_data("../data/model_1_samples.h5", "../data/augmented_model_1_samples.h5", "../data/transformation_matrices.pkl", 4, h=80, w=145, sigma=0.1, random_transformation_n=0, static_transformations=True, static_transformation_n=400, rgb=True, border_value=[244,244,244])
generate_augmented_data("../data/model_2_samples.h5", "../data/augmented_model_2_samples.h5", "../data/transformation_matrices.pkl", 4, h=80, w=145, sigma=0.1, random_transformation_n=0, static_transformations=True, static_transformation_n=400, rgb=True, border_value=[244,244,244])
#generate_augmented_data("../../../../hog_object_detection/positive_samples.h5","../../../../hog_object_detection/positive_augmented_samples.h5", "../data/transformation_matrices.pkl", 2, h=128, w=128, sigma=0.1, random_transformation_n=0, static_transformations=True, rgb=True, static_transformation_n=400, border_value=[244,244,244])#We use =400 so we just get all of them w/e the # is
