#std libs
import json
import time

#external libs
import lira1
from lira1 import *

import config_base

import sample_loader
from sample_loader import *

import output_grapher
from output_grapher import *

import sms_notifications

#General
run_count =                 1#Number of times to run the same config over again to get an accurate representation of it's accuracy and results
epochs =                    800#Number of times we do a full training loop
training_data_subsections=  None#Enable this to get accuracy quicker. Good for initially figuring stuff out and saving time in experimentation

#Percentage of entire data for each subset
p_training_data =           0.80
p_validation_data =         0.10
p_test_data =               0.10

archive_dir = "../data/samples_subs2.pkl.gz"#Dir to get all of our data from
whole_normalization =       True#To normalize on the entire training data. Will add batch normalization soon

#Scheduling
early_stopping=             False#Enable to stop training after our schedule parameter reaches a stop threshold, and fill in the remaining space with last obtained value
automatic_scheduling=       False#Enable to automatically judge the average slope of our validation % over the past scheduler_check_interval epochs and decrease by param_decrease_rate if it is < 0.0
#schedule_exp_rate = None#If this isn't none it will ignore automatic scheduling and automatically decrease every n epochs
scheduler_check_interval =  50#Period to look back for judging if it's time to automatically schedule
param_decrease_rate =       2#Amount to decrease by, i.e. (eta /= param_decrease_rate) if our average slope in our past scheduler_check_interval is < 0.0

#Outputs
output_training_cost=       True
output_training_accuracy=   True
output_validation_accuracy= True
output_test_accuracy=       True

#Ensemble
training_ensemble =         False

#Output
output_title = "Lira MK 1"#Title to show on the graph if graphing output
output_filename = "lira1"#The filename to save our output, and layers if we choose to save net
print_results =             False#Print our accuracies
print_perc_complete =       False#Give a live percentage until completions
update_output =             True#Actually update the output with our new configs. Don't disable this unless you just want to graph again or use pre-existing output
graph_output =              True#Graphs the results with output_grapher.py
save_net =                  True#Save our network layers as <output_filename>.pkl.gz
literal_print_output =      False#Literally print this to a networked printer. Not working (yet)
print_times =               False#Print the elapsed time each config took to train, on average if run_count > 1
sms_alerts =                False#Send finished alert to assigned number with sms_notifications.py
subplot_seperate_configs =  False#Seperate the config graphs into different subplots, rather than combining into one for ease of comparison.

#input_dims = 640*1160
#input_dims = 160*290
input_dims = 80*145
output_dims = 5

#Gotta make seperate Network instances for each run else the params don't get re-initialized

'''
Miscellaneous configs that I don't want to have to write again
            [
                [Network([ 
                    ConvPoolLayer(image_shape=(14, 1, 51, 51),
                        filter_shape=(20, 1, 8, 8),
                        poolsize=(2,2)),
                    FullyConnectedLayer(n_in=22*22*20, n_out=2000), 
                    FullyConnectedLayer(n_in=2000, n_out=100), 
                    FullyConnectedLayer(n_in=100, n_out=30), 
                    SoftmaxLayer(n_in=30, n_out=7)], 14), 14, 
                    1.0, 0.0, 0.0, 100, 10, ""] 
                for r in range(run_count)
            ],
            [
                [Network([ 
                    FullyConnectedLayer(n_in=input_dims, n_out=300, activation_fn=sigmoid), 
                    FullyConnectedLayer(n_in=300, n_out=80, activation_fn=sigmoid), 
                    FullyConnectedLayer(n_in=80, n_out=20, activation_fn=sigmoid), 
                    SoftmaxLayer(n_in=20, n_out=output_dims)], 10, cost=log_likelihood), 10, 
                    1.0, 'momentum', 0.0, 0.0, 0.0, 0.0, 50, 2, "Initial tests"] 
                for r in range(run_count)
           ],
            [
                [Network([ 
                    ConvPoolLayer(image_shape=(10, 1, 160, 290),
                        filter_shape=(20, 1, 13, 23),
                        poolsize=(2,2)),
                    ConvPoolLayer(image_shape=(10, 20, 74, 134),
                        filter_shape=(40, 20, 13, 23),
                        poolsize=(2,2)),
                    FullyConnectedLayer(n_in=31*56*40, n_out=2000), 
                    FullyConnectedLayer(n_in=2000, n_out=100), 
                    SoftmaxLayer(n_in=100, n_out=5)], 10), 10, 
                    1.0, 'momentum', 0.0, 0.0, 0.0, 0.0, 100, 10, "Replicating the experiment with reasonable settings"] 
                for r in range(run_count)
            ],
            [
                [Network([ 
                    ConvPoolLayer(image_shape=(11, 1, 80, 145),
                        filter_shape=(20, 1, 7, 12),
                        poolsize=(2,2),
                        poolmode='max',
                        activation_fn=sigmoid),
                    FullyConnectedLayer(n_in=20*37*67, n_out=1000), 
                    FullyConnectedLayer(n_in=1000, n_out=100), 
                    SoftmaxLayer(n_in=100, n_out=5)], 11), 11, 
                    0.28, 'momentum', 0.0, 0.0, 0.0, -0.02, 100, 10, "Sigmoid"] 
                for r in range(run_count)
            ],
            [
                [Network([ 
                    ConvPoolLayer(image_shape=(11, 1, 80, 145),
                        filter_shape=(20, 1, 7, 12),
                        poolsize=(2,2),
                        poolmode='max',
                        activation_fn=tanh),
                    FullyConnectedLayer(n_in=20*37*67, n_out=1000, activation_fn=tanh), 
                    FullyConnectedLayer(n_in=1000, n_out=100, activation_fn=tanh), 
                    SoftmaxLayer(n_in=100, n_out=5)], 11), 11, 
                    0.28, 'momentum', 0.0, 0.0, 0.0, -0.02, 100, 10, "Tanh"] 
                for r in range(run_count)
            ],
            [
                [Network([ 
                    ConvPoolLayer(image_shape=(11, 1, 80, 145),
                        filter_shape=(20, 1, 7, 12),
                        poolsize=(2,2),
                        poolmode='max',
                        activation_fn=ReLU),
                    FullyConnectedLayer(n_in=20*37*67, n_out=1000, activation_fn=ReLU), 
                    FullyConnectedLayer(n_in=1000, n_out=100, activation_fn=ReLU), 
                    SoftmaxLayer(n_in=100, n_out=5)], 11), 11, 
                    0.28, 'momentum', 0.0, 0.0, 0.0, -0.02, 100, 10, "ReLU"] 
                for r in range(run_count)
            ],
            [
                [Network([ 
                    ConvPoolLayer(image_shape=(11, 1, 80, 145),
                        filter_shape=(20, 1, 7, 12),
                        poolsize=(2,2),
                        poolmode='max', activation_fn=ReLU),
                    FullyConnectedLayer(n_in=20*37*67, n_out=1000, p_dropout=0.75, activation_fn=ReLU), 
                    FullyConnectedLayer(n_in=1000, n_out=100, p_dropout=0.75, activation_fn=ReLU), 
                    SoftmaxLayer(n_in=100, n_out=5, p_dropout=0.75)], 11), 11, 
                    0.28, 'vanilla', 0.0, 0.0, 0.1, -3.0/epochs, 100, 10, "My experimental light CDNN topology"] 
                for r in range(run_count)
            ],
            [
                [Network([ 
                    ConvPoolLayer(image_shape=(11, 1, 80, 145),
                        filter_shape=(20, 1, 7, 12),
                        poolsize=(2,2),
                        poolmode='max', activation_fn=sigmoid),
                    ConvPoolLayer(image_shape=(11, 20, 37, 67),
                        filter_shape=(40, 20, 6, 10),
                        poolsize=(2,2),
                        poolmode='max', activation_fn=sigmoid),
                    ConvPoolLayer(image_shape=(11, 40, 16, 29),
                        filter_shape=(80, 40, 3, 4),
                        poolsize=(2,2),
                        poolmode='max', activation_fn=sigmoid),
                    FullyConnectedLayer(n_in=80*7*13, n_out=1000, p_dropout=0.825, activation_fn=sigmoid), 
                    FullyConnectedLayer(n_in=1000, n_out=100, p_dropout=0.825, activation_fn=sigmoid), 
                    SoftmaxLayer(n_in=100, n_out=5, p_dropout=0.825)], 11), 11, 
                    0.28, 'vanilla', 0.0, 0.0, 1e-5, -3.0/epochs, 100, 10, "CDNN topology #2, the bigger, deeper topology"] 
                for r in range(run_count)
            ],
'''
configs = [
            [
                [Network([ 
                    ConvPoolLayer(image_shape=(11, 1, 80, 145),
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
                for r in range(run_count)
            ],
        ]

config_count = len(configs)
output_types = config_base.get_output_types(output_training_cost, output_training_accuracy, output_validation_accuracy, output_test_accuracy)

if update_output:
    #First, we run our configurations
    f = open('{0}_output.txt'.format(output_filename), 'w').close()
    if training_ensemble: ensemble_nets = []
    output_dict = {}
    config_times = []
    
    for config_index, config in enumerate(configs):
        config_start = time.time()
        for run_index in range(run_count): 

            #Do this every run so as to avoid any accidental bias that might arise
            training_data, validation_data, test_data = get_data_subsets(p_training = p_training_data, p_validation = p_validation_data, p_test = p_test_data, archive_dir=archive_dir)
            training_data, validation_data, test_data, normalize_data = load_data_shared(training_data=training_data, validation_data=validation_data, test_data=test_data, normalize_x=whole_normalization)

            output_dict[run_index] = {}
            net = config[run_index][0]
            net.output_config(
                output_filename=output_filename, 
                training_data_subsections=training_data_subsections, 
                early_stopping=early_stopping,
                automatic_scheduling=automatic_scheduling,
                output_training_cost=output_training_cost,
                output_training_accuracy=output_training_accuracy,
                output_validation_accuracy=output_validation_accuracy,
                output_test_accuracy=output_test_accuracy,
                print_results=print_results,
                print_perc_complete=print_perc_complete,
                config_index=config_index,
                config_count=config_count,
                run_index=run_index,
                run_count=run_count,
                output_types=output_types)

            #For clarity
            mini_batch_size = config[run_index][1]
            learning_rate = config[run_index][2]
            optimization = config[run_index][3]
            optimization_term1 = config[run_index][4]
            optimization_term2 = config[run_index][5]
            regularization_rate = config[run_index][6]
            schedule_exp_rate = config[run_index][7]
            scheduler_check_interval = config[run_index][8]
            param_decrease_rate = config[run_index][9]
            output_dict = net.SGD(output_dict, training_data, epochs, mini_batch_size, learning_rate, validation_data, test_data, optimization=optimization, optimization_term1=optimization_term1, optimization_term2=optimization_term2, lmbda=regularization_rate, 
                    schedule_exp_rate=schedule_exp_rate, scheduler_check_interval=scheduler_check_interval, param_decrease_rate=param_decrease_rate)

            if training_ensemble:
                ensemble_nets.append(Network(net.layers, config_base.size(test_data)))
        
        #After all runs have executed
        config_time = time.time() - config_start
        config_times.append(config_time)

        #If there were more than one runs for this configuration, we average them all together for a new one
        #We do this by looping through all our y values for each epoch, and doing our usual mean calculations
        if run_count > 1:
            output_dict = config_base.get_avg_run(output_dict, epochs, run_count, output_types)

        #Write the output for this config's runs
        f = open('{0}_output.txt'.format(output_filename), 'a')
        f.write(json.dumps(output_dict))
        #add a newline to seperate our configs
        f.write("\n")
        #wrap up by closing our file behind us.
        f.close()

    #Save layers
    if save_net:
        config_base.save_net(net, output_filename, normalize_data, input_dims)

    for config_index, config_time in enumerate(config_times):
        print "Config %i averaged %f seconds" % (config_index, config_time / float(run_count))

    if sms_alerts:
        sms_notifications.send_sms("\nMain Configuration Finished.")

    if training_ensemble:
        print "Training Accuracy: %f" % config_base.get_ensemble_accuracy(training_accuracy)
        print "Validation Accuracy: %f" % config_base.get_ensemble_accuracy(validation_data)
        print "Test Accuracy: %f" % config_base.get_ensemble_accuracy(test_data)
            

if graph_output:
    output_grapher = OutputGrapher(output_title, output_filename, configs, epochs, run_count, output_types, output_training_cost, output_training_accuracy, output_validation_accuracy, output_test_accuracy, subplot_seperate_configs, print_times, update_output)
    output_grapher.graph()
