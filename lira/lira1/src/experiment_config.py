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

#Global config settings
def train_model(nn):
    global_config = {
        'input_dims': [80, 145],
        'output_dims': 7,#We have added another class, so you should not train the model until you've generated a new archive with the live_samples.h5 archive, which should contain the new classification's training data.
        'run_count': 1,
        'epochs': 40,
        'archive_dir': os.path.expanduser("~/programming/machine_learning/tuberculosis_project/lira/lira1/data/samples.h5"),
        'p_training': 0.7,
        'p_validation': 0.15,
        'p_test': 0.15,
        'lira_data': True,
        'output_title': nn,
        'graph_output': True,
        'update_output': True,
        'subplot_seperate_models': False,
        'print_times': False,
        'save_net': True,
        'output_cost' : True,
        'output_training_accuracy' : True,
        'output_validation_accuracy' : True,
        'output_test_accuracy' : True,
        'bbho_optimizing': False,
    }

    #Models to compare
    """
                        [
                            [
                                [Sequential(),
                                Convolution2D(20, 7, 12, border_mode="valid", input_shape=(80, 145, 1), W_regularizer=l2(1.77828e-5)),
                                    Activation("sigmoid"),
                                    MaxPooling2D(),

                                Convolution2D(40, 6, 10, border_mode="valid", W_regularizer=l2(1.77828e-5)),
                                    Activation("sigmoid"),
                                    MaxPooling2D(),

                                Flatten(),

                                Dense(1024, W_regularizer=l2(1.77828e-5)),
                                    Activation("sigmoid"),

                                Dense(100, W_regularizer=l2(1.77828e-5)),
                                    Activation("sigmoid"),

                                #Dropout(0.0),

                                Dense(global_config["output_dims"], W_regularizer=l2(1.77828e-5)),
                                    Activation("softmax")],

                                {    
                                    'cost': "categorical_crossentropy", 
                                    'mb_n': 74,
                                    'optimizer': Adam(1e-4),
                                    'data_normalization': True,
                                    'label': "",
                                }
                            ] for run in range(global_config["run_count"])
                        ],
    """
    model_configs = [
                        [
                            [
                                [Sequential(),
                                Convolution2D(20, 7, 12, border_mode="valid", input_shape=(80, 145, 1), W_regularizer=l2(0.0158489)),
                                    Activation("tanh"),
                                    MaxPooling2D(),

                                Convolution2D(40, 6, 10, border_mode="valid", W_regularizer=l2(0.0158489)),
                                    Activation("tanh"),
                                    MaxPooling2D(),

                                Flatten(),

                                Dense(1024, W_regularizer=l2(0.0158489)),
                                    Activation("tanh"),

                                Dense(100, W_regularizer=l2(0.0158489)),
                                    Activation("tanh"),

                                Dropout(0.92),

                                Dense(global_config["output_dims"], W_regularizer=l2(0.0158489)),
                                    Activation("softmax")],

                                {    
                                    'cost': "categorical_crossentropy", 
                                    'mb_n': 44,
                                    'optimizer': Adam(1e-4),
                                    'data_normalization': True,
                                    'label': "",
                                }
                            ] for run in range(global_config["run_count"])
                        ],
                    ]
    """
    [
        [
            [Sequential(),
            Convolution2D(20, 7, 12, border_mode="valid", input_shape=(80, 145, 1), W_regularizer=l2(1e-4)),
                Activation("sigmoid"),
                MaxPooling2D(),

            Convolution2D(40, 6, 10, border_mode="valid", W_regularizer=l2(1e-4)),
                Activation("sigmoid"),
                MaxPooling2D(),

            Flatten(),

            Dense(1024, W_regularizer=l2(1e-4)),
                Activation("sigmoid"),

            Dense(100, W_regularizer=l2(1e-4)),
                Activation("sigmoid"),

            Dropout(.175),

            Dense(global_config["output_dims"], W_regularizer=l2(1e-4)),
                Activation("softmax")],

            {    
                'cost': "categorical_crossentropy", 
                'mb_n': 50,
                'optimizer': Adam(1e-4),
                'data_normalization': True,
                'label': "Original Parameters",
            }
        ] for run in range(global_config["run_count"])
    ],
    """

    #Handle our models with DENNIS MK 6
    output_dict = handle_models(global_config, model_configs)
