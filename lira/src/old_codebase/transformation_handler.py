"""
Functions exclusively for the purpose of dealing with affine transformations,
    from generating training data, to applying transformations, to anything else.

I would use other libraries (probably this one https://github.com/aleju/imgaug) if I didn't need to create my own transformation matrices / mechanisms for applying the transformations to our data.
    Fortunately, I don't need to include all the code for applying transformations here, as we can do that with OpenCV's WarpAffine() function.

This makes things much easier, and shortens the code in this file a great deal.

-Blake Edwards / Dark Element
"""
import numpy as np
import cv2
import sys

def generate_2d_transformed_data(data, sigma=0.1, transformation_n=9, static_transformation_matrices=[], border_value=0):
    """
    Randomly generate them if we have a number to make, then add our specific ones if we passed them in
    Arguments:
        data: 
            A np array of shape (N, H, W, C),
                N: Number of samples in the data
                H: Height of each sample
                W: Width of each sample
                C: Channels in each sample

            This is the data to have affine transformations applied, transformed, and returned with.
            Be careful, if you pass in 60,000 images, with transformation_n=9, you will end up with 10*60,000 = 600,000 result images.
            This can also be thought of as the X argument in a normal dataset.

        sigma: 
            Defaults to 0.1, this is a parameter to control how large the affine transformations will be when randomly generating.
            Our formula where this is used is as follows:

                [1 0 0]           [r r r]
            T = [0 1 0] + sigma * [r r r]
                [0 0 1]           [0 0 0]

            Where:
                T is each transformation matrix,
                and each r is drawn from the standard normal distribution.

        transformation_n:
            Defaults to 9, so that if each sample in the data has 9 transformed versions of it created, the data size will increase by 10x as a result.
            Used with sigma, this is the number of randomly generated transformation matrices to create, in the random generation process.
            Set this to 0 if you don't want to randomly generate any. 

        static_transformation_matrices: A np array of shape (N, 3, 3),
            N: Number of transformation matrices
            and 3x3 for each 2d transformation matrix
            
            If not supplied, will default to using the `sigma` argument and `transformation_n` arguments.
            If supplied, will append transformation matrices to randomly generated transformation matrices. 
            If you only want to use these, make sure to set transformation_n to 0 !
            Only supply this if you want to use your own specific transformation matrices.

        Warning: transformation_n or static_transformation_matrices must have a value. Do not set both to 0/[].

        border_value:
            The constant value to pad the edges of our image with after transformation, if we have moved part of it out of the viewport.
            Defaults to 0, padding with black.

        Returns:
            transformed_data:
                A np array of shape (N*(transformation_n+1), H, W, C),
                    With all dimensions the same as data, except for the first dimension, which is now N*(transformation_n+1)
                        due to the amount of transformed versions of the original input which are created, plus the original input.
                    The samples here retain the order of the original data argument, s.t.:
                        if data[0] is a picture of the number 7 (i.e. with label 7), 
                            transformed_data[0:transformation_n+2] will also be labelled as 7s, 
                            with transformed_data[0] being the original 7, 
                            and with transformed_data[1:transformation_n+2] being the transformed 7s.
                        We essentially put an identity transformation at the beginning of our transformation matrices, since we have the original data sample as the first entry in each group of transformed samples.
                For an example of 4 transformation matrices, the resulting data will be like the following example:
                    [data[0], transformation1(data[0]), transformation2(data[0]), transformation3(data[0]), transformation4(data[0]), data[1], transformation1(data[1]), transformation2(data[1]), ...]
                So that the data retains it's original order, with each transformation of the data repeating after the original sample transformation_n times.

            transformation_matrices:
                A np array of shape (transformation_n, 3, 3),
                    containing the full array of randomly generated transformation matrices if this function was called with sigma and transformation_n,
                    along with our static, given transformation matrices appended to the end of that, if there are any provided.
    """
    if transformation_n == 0 and static_transformation_matrices == []:
        """
        One of these needs to be set, and if neither are, then we halt execution by returning.
        """
        return data, []

    if transformation_n > 0:
        """
        Randomly generate our transformation matrices using the method discussed above.
        """
        """
        First, we create our (transformation_n, 3, 3) matrices drawn from the standard normal
            by using np.random.randn and then inserting it into the correct location in a zeroed array.
        """
        transformation_matrices = np.zeros((transformation_n, 3, 3))
        transformation_matrices[:, :2, :] = np.random.randn(transformation_n, 2, 3)
        
        """
        Then multiply this by sigma and add all of it to an identity matrix of the same shape.
            We get our identity matrix by making a 3x3 one, then repeating this with python's use of the * operator.
        """
        transformation_matrices *= sigma
        transformation_matrices += np.array([np.eye(3)]*transformation_n) 

    """
    If we have been given static transformation matrices,
    """
    if static_transformation_matrices != []:
        """
        If we have any hand-given, or static, transformation matrices, 
        """
        if transformation_n > 0:
            """
            We append them to our transformation_matrices that were randomly generated, if we have any of those.
            """
            transformation_matrices = list(transformation_matrices)
            transformation_matrices.extend(static_transformation_matrices)
            transformation_matrices = np.array(transformation_matrices)
        else:
            """
            Otherwise, we just set transformation_matrices to our static ones if we don't have any randomly generated transformation matrices
            """
            transformation_matrices = static_transformation_matrices

    """
    Regardless of parameters supplied, we now have transformation matrices (unless both options were set to 0/None).
    Now, we get the length of them to make sure we have the right number regardless of parameter choice.
    """
    n = len(data)
    transformation_n = len(transformation_matrices)

    """
    We then get the individual dimensions of each sample, for use in our affine transformations later.
        Since opencv wants it in format W, H, we have to swap the first two dimensions.
        Since opencv also wants it in a tuple, we then cast it to a tuple.
        Since opencv doesn't want the color, we don't include that.

    Opencv is picky.
    """
    sample_dims = data[0].shape
    sample_dims = [sample_dims[1], sample_dims[0]]
    sample_dims = tuple(sample_dims)

    """
    We then get the dimensions by replacing the first dimension of our data array with n*(transformation_n+1) (since we have always have one identity transformation, in addition to our other transformations)
        which we do via python's list's extend method on all dimensions but the first in our data array
    """
    transformed_data_dims = [n*(transformation_n+1)]
    transformed_data_dims.extend(data.shape[1:])

    """
    Using this, we generate a zeroed np array of our transformed data, since it is going to very likely take up a lot of memory.
        Since it will take up a lot of memory, we also use a np.memmap
    """
    transformed_data = np.memmap("transformed_data.dat", dtype="float32", mode="w+", shape=tuple(transformed_data_dims))

    """
    Then, with all this ready, we loop through each sample in our original data and place it in the transformed data
    """
    for sample_i, sample in enumerate(data):
        for transformation_i in range(transformation_n+1):
            """
            Loop through transformation_n + 1, so that we can use 0 for our original data, 
                and the rest for referencing the transformation matrix to use.

            If it helps, think of the 0th as the identity transformation.
            """
            """
            Get our index in our transformed data array via
                (transformation_n+1) * sample_i + transformation_i
            Since we have to use +1 for taking our original data / identity transformation into account,
                then we can treat this problem as if we were getting the indices in a flattened vector from the indices in a matrix of shape (n, transformation_n)
                which we would normally do as index = width * row_i + col_i
            """
            transformed_sample_i = (transformation_n+1) * sample_i + transformation_i

            if transformation_i == 0:
                """
                If this is the first transformation, we don't transform and we just insert original element
                """
                transformed_data[transformed_sample_i] = sample
            else:
                """
                Otherwise, we subtract one from our transformation_i (Since we have our identity transformation added),
                    then get our transformation matrix.
                """
                transformation_matrix = transformation_matrices[transformation_i-1]
                
                """
                We then transform this sample, using opencv's warp affine and setting:
                    the output equal to the original input size,
                    and the borderValue equal to the argument passed in.
                """
                transformed_sample = cv2.warpAffine(sample, transformation_matrix[:2], sample_dims, borderValue=border_value)

                """
                Then we put our new transformed sample into the corresponding location in our transformed_data array.
                """
                transformed_data[transformed_sample_i] = transformed_sample

    """
    We then return our transformed_data array, and transformation_matrices.
    """
    return transformed_data, transformation_matrices

def generate_transformed_references(labels, transformation_n):
    """
    Arguments:
        labels:
            A np array of same size as data argument to generate_2d_transformed_data(), 
                so that the shape is (N, ...)
                with the remaining dimensions irrelevant.
            This contains the labels for the original data, and will be used to generate labels for the transformed data.

        transformation_n:
            Our number of transformations applied on the original data.

    Returns:
        transformed_labels:
            A np array of shape (N*(transformation_n+1), ...)
                With each label corresponding to transformed_data, instead of the original data.
                Assumes that generate_2d_transformed_data(), or one of our functions for generating transformed data,
                    was used to generate the transformed data. 
                If 4 matrices were supplied, result will look like:
                    [label1, label1, label1, label1, label1, label2, label2, ...]
                Since it repeats our label for each transformation we apply.

        Be careful when passing in different data than that returned by generate_2d_transformed_data(),
            as this may cause errors further down your pipeline.
        
        For more documentation, view https://docs.scipy.org/doc/numpy/reference/generated/numpy.repeat.html
    """
    return np.repeat(labels, transformation_n+1)
