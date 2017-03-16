"""
Functions for graphing LIRA's results.

-Blake Edwards / Dark Element
"""
    
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

def graph_subplot(result, result_i, result_str):
    """
    Arguments:
        result: a np array of shape (run_count, epochs, 1)
            which is a subset of our results array, and contains only one metric / result.

        result_i: the index of this result, will be used to determine which subplot to put our result into.

    Returns:
        Places the output over epochs in each run onto the subplot as dotted lines, 
            then computes an average run over epochs and places that as a solid line.
    """
    """
    Get run_count and epochs
    """
    run_count, epochs = result.shape[0], result.shape[1]

    """
    Get x axis via epochs
    """
    x_axis = np.linspace(0, epochs, epochs)

    """
    Get the correct subplot index and title
    """
    plt.subplot(2,2,result_i+1)
    plt.title(result_str)

    """
    Loop through independent runs and place on subplot
    """
    for run_i in range(run_count):
        plt.plot(x_axis, result[run_i], ls='--')

    """
    Place final averaged run on subplot, and exit.
    """
    plt.plot(x_axis, np.mean(result, axis=0), lw=2.0, c='blue')

        
def graph_results(results):
    """
    Arguments:
        results: a np array of shape (run_count, epochs, 4)
            where each entry on the zeroth axis, run_count, contains the results of an independently run model,
            each entry on the first axis, epochs, contains the results of each model at each epoch,
            and the second axis is training loss, training accuracy, validation accuracy, and test accuracy
                (0 < accuracy < 1, multiply by 100 to get percentage)

    Returns:
        With 4 subplots, graphs dotted lines for the outputs of each run, 
            then computes the average of all runs and graphs that as a solid blue line.
        Returns no values.
    """

    print "Graphing Results..."

    """
    So we first get these values from the initial array
    """
    run_count, epochs = results.shape[0], results.shape[1]
    
    """
    Then we need our training loss, training accuracy, validation accuracy, and test accuracy in separate arrays.
    """
    training_loss, training_acc, validation_acc, test_acc = np.split(results, 4, axis=2)

    """
    Graph each metric/result on a separate subplot.
    """
    graph_subplot(training_loss, 0, "Training Loss")
    graph_subplot(training_acc, 1, "Training Accuracy")
    graph_subplot(validation_acc, 2, "Validation Accuracy")
    graph_subplot(test_acc, 3, "Test Accuracy")

    """
    Show our plot.
    """
    plt.show()

"""
r = 31
results = np.concatenate((np.ones((r,30,1)), np.ones((r,30,1))*2, np.ones((r,30,1))*3, np.ones((r,30,1))*4), axis=2)
results = results + np.random.randn(r,30,4)/2
graph_results(results)
"""
