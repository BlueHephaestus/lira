import os, sys
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2

sys.path.append(os.path.expanduser("~/programming/machine_learning/dennis/dennis6/src"))

import dennis6
from dennis6 import handle_models

import dataset_obj
from dataset_obj import *

import lira_optimization_input_handler
from lira_optimization_input_handler import handle_raw_hps

#Global config settings
global_config= {
    'input_dims': [80,145],
    'output_dims': 6,
    'run_count': 3,
    'epochs': 40,
    'archive_dir': os.path.expanduser("~/programming/machine_learning/tuberculosis_project/lira/lira1/data/samples.h5"),
    'raw_rim_dir': os.path.expanduser("~/programming/machine_learning/tuberculosis_project/lira/lira1/data/samples/raw_rim"),
    'p_training': 0.7,
    'p_validation': 0.15,
    'p_test': 0.15,
    'lira_data': True,
    'subsection_n': 64,
    'output_title': "BBHO Optimizations",
    'graph_output': False,
    'update_output': True,
    'subplot_seperate_models': False,
    'print_times': False,
    'save_net': False,
    'output_cost' : True,
    'output_training_accuracy' : True,
    'output_validation_accuracy' : True,
    'output_test_accuracy' : False,
    'bbho_optimizing': True,
}

#Global config settings
class Configurer(object):

    def __init__(self, epochs, run_count):
        self.epochs = epochs#Have to have this here since it needs to be the same across configs
        self.run_count = run_count 

        #With our values from BBHO, reassign these
        global_config["epochs"] = self.epochs
        global_config["run_count"] = self.run_count

        if global_config['bbho_optimizing']:
            output_filename = "%s_optimizations.txt" % global_config["output_title"].lower().replace(" ", "_")
            self.f = open(output_filename, "w")

    def run_config(self, hps):

        if global_config['bbho_optimizing']:
            mb_n, regularization_rate, dropout_perc, activation_fn, cost, hp_str = handle_raw_hps(hps)
            self.f.write(hp_str)
        
        model_configs = [
                            [
                                [
                                    [Sequential(),
                                    Convolution2D(20, 7, 12, border_mode="valid", input_shape=(80, 145, 1), W_regularizer=l2(regularization_rate)),
                                        Activation(activation_fn),
                                        MaxPooling2D(),

                                    Convolution2D(40, 6, 10, border_mode="valid", W_regularizer=l2(regularization_rate)),
                                        Activation(activation_fn),
                                        MaxPooling2D(),

                                    Flatten(),

                                    Dense(1024, W_regularizer=l2(regularization_rate)),
                                        Activation(activation_fn),

                                    Dense(100, W_regularizer=l2(regularization_rate)),
                                        Activation(activation_fn),

                                    Dropout(dropout_perc),

                                    Dense(global_config["output_dims"], W_regularizer=l2(regularization_rate)),
                                        Activation("softmax")],

                                    {    
                                        'cost': cost, 
                                        'mb_n': mb_n,
                                        'optimizer': Adam(1e-4),
                                        'data_normalization': True,
                                        'label': "",
                                    }
                                ] for run in range(global_config["run_count"])
                            ]
                        ]

        #Handle our models with DENNIS MK 6
        output_dict = handle_models(global_config, model_configs)

        #With our output_dict, get our average validation accuracies by doing:
        #For each epoch in # epochs, get average value from average run entry in output dict
        if global_config["run_count"] > 1:
            avg_val_accs = np.array([output_dict[global_config["run_count"]+1][avg_epoch][2] for avg_epoch in range(global_config["epochs"])])
        else:
            avg_val_accs = np.array([output_dict[0][avg_epoch][2] for avg_epoch in range(global_config["epochs"])])

        return avg_val_accs

