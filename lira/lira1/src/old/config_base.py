import numpy as np

def get_output_types(output_training_cost, output_training_accuracy, output_validation_accuracy, output_test_accuracy):
    output_types = 0
    if output_training_cost:
        output_types += 1
    if output_training_accuracy:
        output_types += 1
    if output_validation_accuracy:
        output_types += 1
    if output_test_accuracy:
        output_types += 1
    return output_types

def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]

def get_avg_run(output_dict, epochs, run_count, output_types):
    output_dict[run_count+1] = {}#For our new average entry
    for j in range(epochs):
        output_dict[run_count+1][j] = []#For our new average entry
        for o in range(output_types):
            avg = sum([output_dict[r][j][o] for r in range(run_count)]) / run_count
            output_dict[run_count+1][j].append(avg)
    return output_dict

def save_net(net, output_filename, normalize_data, input_dims):
    print "Saving Neural Network Layers..."
    net.save('../saved_networks/%s.pkl.gz' % output_filename)
    f = open('../saved_networks/%s_metadata.txt' % (output_filename), 'w')
    f.write("{0}\n{1}\n{2}".format(normalize_data[0], normalize_data[1], input_dims))
    f.close()

def get_ensemble_accuracy(ensemble_nets, data):
    #Get the accuracy of our ensemble net by looping through each sample and looping through each net
    #validation_x, validation_y = validation_data
    ensemble_predictions = []
    for ensemble_net in ensemble_nets:
        ensemble_predictions.append(ensemble_net.predict(data))
    #predictions is array of predictions for each ensemble net on all the validation_data, so we get the mode of the 
    #Get most common output, mode
    ensemble_choices = [np.bincount(ensemble_votes).argmax() for ensemble_votes in np.array(ensemble_predictions).transpose()]#Loop through each column and get the mode
    x, y = data

    return T.mean(T.eq(ensemble_choices, y)).eval()
