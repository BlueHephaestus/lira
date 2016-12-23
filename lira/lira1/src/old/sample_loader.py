import librosa
import numpy as np
import os
import cPickle, gzip #for storing our data 
import theano
import theano.tensor as T
#Get our big array of data and seperate it into training, validation, and test data
#According to the percentages passed in

##Defaults:
#80% - Training
#10% - Validation
#10% - Test
def unison_shuffle(a, b):
  rng_state = np.random.get_state()
  np.random.shuffle(a)
  np.random.set_state(rng_state)
  np.random.shuffle(b)

def get_data_subsets(archive_dir="../data/mfcc_samples.pkl.gz", p_training=0.8, p_validation=0.1, p_test=0.1):
  print "Getting Training, Validation, and Test Data..."
  
  f = gzip.open(archive_dir, 'rb')
  data = cPickle.load(f)

  """
  The following is specific to LIRA implementation:
    We have our X as shape (95*64, 80*145), and 
    our Y as shape (95*64,)

    This is because we have 95 full images, and since we 
    divide them into 80x145 subsections, we end up with 
    64 subsections for each image, hence 95*64.

  However, Professor Asa Ben-Hur raised a good point in that
    if I am just mixing them up like this, then I could have one of my 
    test data samples(a subsection, remember) be a subsection right 
    next to a training data sample, quite easily. This would ruin
    the validity of our validation and test data, so here's the plan to
    fix that:
 
  We instead split our (95*64, 80*145) matrix into 95 parts, so that we end up 
    with a 3d array of shape (95, 64, 80*145). (Do the same for Y but with a 
    resulting 2d array from the vector) Then, we can do our unison shuffle on the 
    entire images instead of our massive subsection collage.

  This way, we have entire different images for test and validation data than our
    training data, and we ensure valid results when testing.

  With that said,
  BEGIN LIRA SPECIFIC IMAGE STUFF
  """

  n_samples = 95
  data[0] = np.split(data[0], n_samples)
  data[1] = np.split(data[1], n_samples)
  "END LIRA STUFF FOR NOW"

  "USUAL STUFF"
  #n_samples = len(data[0])
  "END USUAL STUFF"

  training_data = [[], []]
  validation_data = [[], []]
  test_data = [[], []]

  n_training_subset = np.floor(p_training*n_samples)
  n_validation_subset = np.floor(p_validation*n_samples)
  #Assign this to it's respective percentage and whatever is left
  n_test_subset = n_samples - n_training_subset - n_validation_subset

  #Shuffle while retaining element correspondence
  print "Shuffling data..."
  unison_shuffle(data[0], data[1])

  #Get actual subsets
  data_x_subsets = np.split(data[0], [n_training_subset, n_training_subset+n_validation_subset])#basically the lines we cut to get our 3 subsections
  data_y_subsets = np.split(data[1], [n_training_subset, n_training_subset+n_validation_subset])

  training_data[0] = data_x_subsets[0]
  validation_data[0] = data_x_subsets[1]
  test_data[0] = data_x_subsets[2]

  training_data[1] = data_y_subsets[0]
  validation_data[1] = data_y_subsets[1]
  test_data[1] = data_y_subsets[2]
  
  "MORE LIRA SPECIFIC STUFF"
  #Now that we've shuffled and split relative to images(instead of subsections), collapse back to matrix (or vector if Y)
  #We do -1 so that it infers we want to combine the first two dimensions, and we have the last argument because we
  #want it to keep the same last dimension. repeat this for all of the subsets
  #Since Y's are just vectors, we can easily just flatten
  training_data[0] = training_data[0].reshape(-1, training_data[0].shape[-1])
  training_data[1] = training_data[1].flatten()
  validation_data[0] = validation_data[0].reshape(-1, validation_data[0].shape[-1])
  validation_data[1] = validation_data[1].flatten()
  test_data[0] = test_data[0].reshape(-1, test_data[0].shape[-1])
  test_data[1] = test_data[1].flatten()

  #print training_data[0].shape, training_data[1].shape, validation_data[0].shape, validation_data[1].shape, test_data[0].shape, test_data[1].shape
  print "# of Samples per subset:"
  print "\t{}".format(training_data[0].shape[0]/64)
  print "\t{}".format(validation_data[0].shape[0]/64)
  print "\t{}".format(test_data[0].shape[0]/64)

  print "Check to make sure these have all the different classes"
  print validation_data[1]
  print list(test_data[1])
  "END MORE LIRA SPECIFIC STUFF"

  return training_data, validation_data, test_data


def generate_input_normalizer(training_data):
    print "Generating Input Normalizer..."
    #we initialize our inputs with a normal distribution - this works by generating a normal distribution based on the mean and standard deviation of our training data since it should be a reasonable way to generalize for test data and so on. It helps to make it a normal distribution so that we can most of the time keep our neurons from saturating straight off, just as we do with weights and biases. Just needed to write this out to make sure I gots it
    #See our written notes
    '''The following line is basically:
    for sample in training_data[0]:
        for frame in sample:
            return frame
    '''
    input_x = [frame for sample in training_data[0] for frame in sample]#for sample in x: for frame in x: return frame
    mean = sum(input_x)/float(len(input_x))
    stddev = np.linalg.norm(input_x-mean)/np.sqrt(len(input_x))
    return mean, stddev

def normalize_input(data, mean, stddev):
    print "Normalizing Input..."
    data[0] = data[0]*stddev + mean
    return data

#### Load the data
def load_data_shared(training_data=None, validation_data=None, test_data=None, normalize_x=False, experimental_dir=None):
    print "Initializing Shared Variables..."
    if not training_data and not experimental_dir:
        #Configuration person fucked up
        print "You must supply either data subsets, or choose the experimental directory"
        return
    if experimental_dir:
        f = gzip.open(experimental_dir, 'rb')
        training_data, validation_data, test_data = cPickle.load(f)

    #normalize input data.
    if normalize_x:
        input_normalizer_mean, input_normalizer_stddev = generate_input_normalizer(training_data)
    else:
        input_normalizer_mean = 0
        input_normalizer_stddev = 1

    training_data = normalize_input(training_data, input_normalizer_mean, input_normalizer_stddev)
    validation_data = normalize_input(validation_data, input_normalizer_mean, input_normalizer_stddev)
    test_data = normalize_input(test_data, input_normalizer_mean, input_normalizer_stddev)

    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.

        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    print "Initializing Configuration..."
    return [shared(training_data), shared(validation_data), shared(test_data), [input_normalizer_mean, input_normalizer_stddev]]

#training_data, validation_data, test_data = get_data_subsets(archive_dir="../data/mfcc_expanded_samples.pkl.gz")
#training_data, validation_data, test_data = load_data_shared(training_data, validation_data, test_data, normalize_x=True)

#get_data_subsets(archive_dir = "../data/samples_subs2.pkl.gz")
