import keras

def handle_raw_hps(hps):
    """
    Arguments:
        hps: A list or tuple of different hyper parameters. These will be parsed with knowledge of what each hyper parameter represents.
    Returns:
        mb_n: Returns 1 if mb_n is < 1
        regularization_rate: Returns 10^(-regularization_rate), to scale it exponentially given a linear value.
        dropout_perc: The original dropout percentage
        activation_fn: The activation function in our list of activation functions, referenced by this value as an index.
        cost: The cost function in our list of cost functions, referenced by this value as an index.
        hp_str: A string with labels for each new hyper parameter, and the corresponding new hyper parameter values

    This is very problem-specific, and you should modify this depending on how many parameters you want to optimize. Throughout this file, I have several other parameters that I previously experimented with optimizing for my problem, and I encourage you to do the same for your problems.

    After parsing all the parameters in the ways specified above, a hyper parameter string is generated, and printed.
    """
    mb_n, regularization_rate, dropout_perc = hps
    """
    Handle all of our raw inputs as given by an optimizer, and return the resulting parsed values,
        as well as a string detailing the hyper parameters
    """

    """
    Make sure we don't have a mini batch size of 0, as that would break things and not make sense.
    """
    mb_n = int(mb_n)
    if mb_n < 1:
        mb_n = 1
    
    """
    Put regularization rate through function that converts
        from linear to logarithmic scale according to 10
        f(0) = 1, f(1) = 0.1, f(4) = 1e-4, and so on.

    It only makes intuitive sense that if we want something that gives
        1e-4 when x = 4, 1e-4 = 10^-4 -> 10^-x
    """
    regularization_rate = 10.0**(-regularization_rate)

    """
    We do nothing with dropout percentage.
    """

    """
    For our activation function, cost, and optimizer, we are given an index. We 
        get the respective activation/cost/optimizer from a list according to this index.
    Note: We floor() our indices and convert to integers first
    """
    """
    activation_fn_i = int(activation_fn_i//1)
    cost_i = int(cost_i//1)
    """
    """
    optimizer_i = int(optimizer_i//1)
    """
    
    """
    activation_fns = [
                        "softplus",
                        "softsign",
                        "relu",
                        "tanh",
                        "sigmoid",
                        "hard_sigmoid",
                        "linear"
                    ]
    costs = [
                        "mean_squared_error",
                        "mean_absolute_error",
                        "mean_absolute_percentage_error",
                        "mean_squared_logarithmic_error",
                        "hinge",
                        "squared_hinge",
                        "binary_crossentropy", 
                        "categorical_crossentropy",
            ]

    """
    """
    optimizers = [
                        SGD(),
                        RMSprop(),
                        Adagrad(),
                        Adadelta(),
                        Adam(),
                        Adamax(),
                        Nadam(),
                 ]

    optimizer_strs = [
                        "SGD",
                        "RMSProp",
                        "Adagrad",
                        "Adadelta",
                        "Adam",
                        "Adamax",
                        "Nadam",
                     ]
    """

    """
    activation_fn = activation_fns[activation_fn_i]
    cost = costs[cost_i]
    """
    """
    optimizer = optimizers[optimizer_i]
    optimizer_str = optimizer_strs[optimizer_i]
    """

    """
    Finally, print our resulting hyper parameter string.
    """
    #hp_str = "\nHYPER PARAMETERS: \nMini Batch Size: %f\nRegularization Rate: %f\nDropout Percentage: %f\nActivation: %s\nCost: %s\n\n" % (mb_n, regularization_rate, dropout_perc, activation_fn, cost)
    hp_str = "\nHYPER PARAMETERS: \nMini Batch Size: %f\nRegularization Rate: %f\nDropout Percentage: %f\n\n" % (mb_n, regularization_rate, dropout_perc)
    print hp_str

    """
    And return everything.
    """
    return mb_n, regularization_rate, dropout_perc, hp_str
