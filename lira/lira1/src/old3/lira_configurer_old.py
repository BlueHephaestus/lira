"""
For configuring Dennis/Lira for one configuration, as a form of black box function.
    Currently so I can let BBHO take a crack at it.

-Blake Edwards / Dark Element
"""
import os, sys, json, time
import numpy as np
import tensorflow as tf

sys.path.append(os.path.expanduser("~/programming/machine_learning/dennis/dennis5/src"))

import dennis5
from dennis5 import *

import dennis_base
from dennis_base import *

import dataset_obj
from dataset_obj import *

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

#Data subsets and dataset
archive_dir = os.path.expanduser("~/programming/machine_learning/tuberculosis_project/lira/lira1/data/samples_subs2.pkl.gz")
p_training =    0.75
p_validation =  0.15
p_test =        0.15

output_config = {
    'output_title': "BBHO LIRA Optimizations",
    'graph_output': False,
    'update_output': True,
    'subplot_seperate_configs': False,
    'print_times': False,
    'save_net': False,
    'output_cost' : False,
    'output_training_accuracy' : False,
    'output_validation_accuracy' : True,
    'output_test_accuracy' : False,

}

#Global config settings
class Configurer(object):

    def __init__(self, epochs, run_count):
        self.epochs = epochs#Have to have this here since it needs to be the same across configs
        self.run_count = run_count 

    def run_config(self, mb_n, regularization_rate, keep_prob):
        #Put regularization rate through function that produces the following values:
        #f(1) = 1, f(3) = 0.1, f(5) = 0.01, f(7) = 0.001, etc
        #Since it's much easier to do a range 1-10 and plug it in here at the end
        regularization_rate = (1.0/np.sqrt(10))**(regularization_rate-1)
        print "Mini Batch Size: {}".format(mb_n)
        print "Regularization Rate: {}".format(regularization_rate)
        print "Keep Prob: {}".format(keep_prob)

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

        """
        This is where we make the magic happen for BBHO.

        Since we are only outputting validation accuracy,
            we get that as an array, and return it to the black_box_functions.py file in bbho's dir.
        """
        return np.array([output_dict[self.run_count+1][i] for i in output_dict[self.run_count+1]])
