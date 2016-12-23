"""
For configuring Dennis to run any sort of comparison or experimental configuration.

Currently running on my LIRA image recognition dataset, 
    with the problem-specific manipulation of data in 
    dataset_obj.py and some stuff in sample_loader.py

Feel free to change that accordingly as your data format changes, 
    the necessary dimensions and format are specified there.

-Blake Edwards / Dark Element
"""
import os, sys, json, time, copy
import numpy as np
import tensorflow as tf

sys.path.append(os.path.expanduser("~/programming/machine_learning/dennis/dennis5/src"))

import dennis5
from dennis5 import *

import dennis_base
from dennis_base import *

import dataset_obj
from dataset_obj import *

import output_grapher
from output_grapher import *

import layers
from layers import *

import costs
from costs import *

import weight_inits
from weight_inits import *

import bias_inits
from bias_inits import *

#Network dimensions
input_dims = 80*145
output_dims = 6

#Global config settings
run_count = 3
epochs = 4000#Have to have this here since it needs to be the same across configs

#Data subsets and dataset
archive_dir = os.path.expanduser("~/programming/machine_learning/tuberculosis_project/lira/lira1/data/samples_subs2.pkl.gz")
p_training =    0.7
p_validation =  0.15
p_test =        0.15

output_config = {
    'output_title': "Testing with new Empty Slide Class",
    'graph_output': True,
    'update_output': True,
    'subplot_seperate_configs': False,
    'print_times': False,
    'save_net': True,
    'output_cost' : True,
    'output_training_accuracy' : True,
    'output_validation_accuracy' : True,
    'output_test_accuracy' : True,

}
"""
CONFIG DOCUMENTATION
1. First argument must be list of layers for the network
2. Second argument must be a dictionary of hyper parameters for the network
    input_dims: the input dimensions
    output_dims: the output dimensions
    cost: the cost function
    mb_n: mini batch size
    optimization_type: 
        The type of optimization to use.
        If you want to have as many arguments default as possible, then you can just use optimization_defaults = True for that.
        If you want none of them, then set optimization_defaults = False.
        However, if you want some on and some off, you can pass in the tensorflow optimization object instead of a string here.
        Types: 'gd', 'adadelta', 'adagrad', 'adagrad-da', 'momentum', 'nesterov', 'adam', 'ftrl', 'rmsprop'
    optimization_term1: First optimization term
    optimization_term1_decay_rate: First optimization term's decay rate, usually only used for when this is learning rate.
    optimization_term2: Second optimization term
    optimization_term3: Third optimization term
    optimization_defaults: described under usage with optimization_type.
    regularization_type: keyword string for the type of regularization to use. 
        Currently supports: 'l1', 'l2'
    regularization_rate: The regularization parameter/rate.
    keep_prob: the probability we keep any given neuron, (1-keep_prob) = dropout percentage.
    label: what the line will be labeled on our final graph.



"""
"""
                    [ConvPoolLayer(image_shape=(11, 1, 80, 145),
                        filter_shape=(20, 1, 7, 12),
                        poolsize=(2,2),
                        poolmode='max', activation_fn=sigmoid),
                    ConvPoolLayer(image_shape=(11, 20, 37, 67),
                        filter_shape=(40, 20, 6, 10),
                        poolsize=(2,2),
                        poolmode='max', activation_fn=sigmoid),
                    FullyConnectedLayer(n_in=40*16*29, n_out=1000, p_dropout=0.825, activation_fn=sigmoid), 
                    FullyConnectedLayer(n_in=1000, n_out=100, p_dropout=0.825, activation_fn=sigmoid), 
                    SoftmaxLayer(n_in=100, n_out=5, p_dropout=0.825)], 11), 11, 
                    0.28, 'vanilla', 0.0, 0.0, 1e-4, -3.0/epochs, 100, 10, "CDNN topology #2, the bigger, deeper topology"] 
"""

"""
configs = [
            [
                [SoftmaxLayer(input_dims, output_dims)], 
                {    
                    'input_dims': input_dims,
                    'output_dims': output_dims,
                    'cost': cross_entropy, 
                    'mb_n': 50,
                    'optimization_type': tf.train.AdamOptimizer(1e-4),
                    'optimization_term1': 0.0,
                    'optimization_term1_decay_rate': 0.05,
                    'optimization_term2': 0.0,
                    'optimization_term3': 0.0,
                    'optimization_defaults': True,
                    'regularization_type': 'l2',
                    'regularization_rate': 1e-4,
                    'keep_prob': 0.825,
                    'data_normalization': True,
                    'label': ""
                }
            ],
          ]
"""
configs = [
            [
                [ConvPoolLayer(
                    [7, 12, 1, 20], 
                    [-1, 80, 145, 1], 
                    conv_padding='VALID', 
                    pool_padding='VALID', 
                    activation_fn=tf.nn.sigmoid,
                ), 
                ConvPoolLayer(
                    [6, 10, 20, 40], 
                    [-1, 37, 67, 20], 
                    conv_padding='VALID', 
                    pool_padding='VALID', 
                    activation_fn=tf.nn.sigmoid,
                ),
                FullyConnectedLayer(40*16*29, 1024),
                FullyConnectedLayer(1024, 100),
                SoftmaxLayer(100, output_dims)], 
                {    
                    'input_dims': input_dims,
                    'output_dims': output_dims,
                    'cost': cross_entropy, 
                    'mb_n': 50,
                    'optimization_type': tf.train.AdamOptimizer(1e-4),
                    'optimization_term1': 0.0,
                    'optimization_term1_decay_rate': 0.05,
                    'optimization_term2': 0.0,
                    'optimization_term3': 0.0,
                    'optimization_defaults': True,
                    'regularization_type': 'l2',
                    'regularization_rate': 1e-4,
                    'keep_prob': 0.825,
                    'data_normalization': True,
                    'label': ""
                }
            ],
          ]
#Default Network init values
cost = cross_entropy
weight_init = weight_inits.glorot_bengio
bias_init = bias_inits.standard

nn_layers = copy.deepcopy(configs[0][0])
#Clean up our title to get a filename, convert characters to lowercase and spaces to underscores
output_filename = output_config['output_title'].lower().replace(" ", "_")

#If we're updating output, clear out the old file if it exists.
if output_config['update_output']: open('{0}_output.txt'.format(output_filename), 'w').close()

#Initialize an empty list to store our time-to-execute for each config
config_times = []

#Loop through entire configurations
for config_i, config in enumerate(configs):
    #Initialize a new output_dict to store the results
    #   of running this configuration over our runs
    output_dict = {}

    #Start a new timer for us to record duration of running config
    config_start = time.time()

    #Loop through number of times to test(runs)
    if output_config['update_output']:
        for run_i in range(run_count):

            #Get our hyper parameter dictionary
            hps = config[1]#Hyper Parameters

            #Do this every run so as to avoid any accidental bias that might arise
            #Load our datasets with our lira_data = True and subsection_n = number of subsections we get from our original training images.
            dataset, normalization_data = load_dataset_obj(p_training, p_validation, p_test, archive_dir, output_dims, hps['data_normalization'], lira_data=True, subsection_n=64)
            
            #Print approximate percentage completion
            perc_complete = perc_completion(config_i, run_i, len(configs), run_count)
            print "--------%02f%% PERCENT COMPLETE--------" % perc_complete

            #Get our Network object differently depending on if optional parameters are present
            if 'cost' in hps:
                cost = hps['cost']
            if 'weight_init' in hps:
                weight_init = hps['weight_init']
            if 'bias_init' in hps:
                bias_init = hps['bias_init']

            #Initialize Network
            net = Network(config[0], hps['input_dims'], hps['output_dims'], dataset, cost, weight_init, bias_init)

            #Finally, optimize and store the outputs
            output_dict[run_i] = net.optimize(output_config, epochs, hps['mb_n'], hps['optimization_type'], hps['optimization_term1'], hps['optimization_term1_decay_rate'], hps['optimization_term2'], hps['optimization_term3'], hps['optimization_defaults'], hps['regularization_type'], hps['regularization_rate'], hps['keep_prob'])

        #Record times once all runs have executed
        config_time = time.time() - config_start
        config_times.append(config_time)

        #Get our average extra run if there were >1 runs
        if run_count > 1:#
            output_dict = get_avg_run(output_dict, epochs, run_count, get_output_types_n(output_config))

        #Save our output to the filename
        save_output(output_filename, output_dict)

    #For prettiness, print completion
    if config_i == len(configs)-1: print "--------100%% PERCENT COMPLETE--------" 

if output_config['save_net'] and output_config['update_output']:
    #Save all of our network data
    print "Saving Model..."
    net.save(output_filename, normalization_data[0], normalization_data[1], nn_layers, input_dims, output_dims, cost, weight_init, bias_init)

#Print time duration for each config
for config_i, config_time in enumerate(config_times):
    print "Config %i averaged %f seconds" % (config_i, config_time / float(run_count))

#Graph all our results
if output_config['graph_output']:
    output_grapher = OutputGrapher(output_config, output_filename, configs, epochs, run_count)
    output_grapher.graph()
