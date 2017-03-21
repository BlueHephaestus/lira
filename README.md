## Description

L.I.R.A. (Lesion Image Recognition and Analysis) is a machine learning for image recognition and analysis project made in collaboration with [Colorado State University's Microbiology Research Lab.](http://mrl.colostate.edu/)

This project was made to help identify the frequency of lesions in infected tuberculosis lung slides, however it has been made efficiently, modularly, and with good documentation, in the hopes that others may also find use in it's various tools. In having a quickly automated system for testing tuberculosis drug treatment, treatments can be tested faster and new statistics can be gained for the efficacy of these treatments.

The problem has been approached with a deep convolutional neural network in Python's Keras library, alongside various other techniques in machine learning and neural networks. If I do not include the remaining information and parameters in the research article to be published at the time of this project's completion, I will put them here.

This project/repo has been made public along with a research paper, found [here when it's finished]()

I am manager of the code for this project, as the primary developer of the code for the project and official collaborator with the MRL. Unfortunately, updating this as it was developed publicly is not an option before the paper was published, so much of this was developed and replaced on my own machines. Because of that, much of the history of the development of this project is not available, and a large amount of it is merely marked under it's own "Initial Commit". Github Premium was purchased at this point, so that I could continue updating the remaining history as it was finished, and then make the repository public when the time came.

The dataset is too large to store here, however we will set up a system for ease of access to others.

We hope that the code here has been documented and designed well enough that many people may modify it for their own uses, but if you have trouble with any of it, don't hesitate to contact the manager of the code for this project (Blake Edwards / Dark Element).

The project is divided into several directories, described below. Every program in these directories (with the exception of LIRA MK1) contains it's own detailed documentation, as well as every function in each of these programs. If anything is not clear enough, I would greatly appreciate any questions or helpful comments. 

#### lira/

The main section of the project, this directory contains the data for training the Keras model (currently a deep convolutional neural network) for lesion classification.

##### MK 1

  This version of LIRA was eased by use of separate bots [DENNIS](https://github.com/DarkElement75/dennis) for the deep learning system it referenced in configuration, and [BBHO](https://github.com/DarkElement75/bbho) for Bayesian Optimization of that configuration. experiment_config.py is a configuration file for DENNIS MK 6, and lira_optimization_input_handler.py and lira_configurer.py are configuration files for use with BBHO.

  The remaining files are used with getting the data formatted so that DENNIS MK 6 can use it to train, and for creating archives that will be useful later. There are some older files and deprecated data handlers as well.

##### MK 2 - LIRA Harder
  
  This version was made to be much more compact, readable, and completely independent of DENNIS. With DENNIS, we fell into [the generalization problem](https://xkcd.com/974/), where at the cost of efficiency/readability I made a system that had tons of options, and could handle more experiments. Unfortunately, this is pointless when I can now create a simple file (as I have now done) with Keras for any problem, with all the customisability of Keras (i.e. a lot).I've now done that with LIRA MK 2; it now minimally exists in a much simpler lira2.py file. It also has numerous other useful features detailed below, to help with model development and testing.

Features:
  1. Simpler, more compact, more modular, and more documented dataset handler for handling our training, validation, and test datasets.
  2. Numerous independent training iterations for random cross-validation, to increase confidence in true model performance.
  3. Saving of model results for future reference.
  4. Saving of model for future use.
  5. Saving of model metadata for future use.
  6. Simpler and more compact graphing of training loss, training accuracy, validation accuracy, and test accuracy across models, with an average result for each across models.

*I have not yet added BBHO compatibility, will come in the future.

#### slide_testing/

  This section of the project is in charge of classifying our test slides, using our network once it is finished training. All of the files here are devoted to loading the network, generating results, and storing and prettifying those results. 

#### slide_labeller/

  This section of the project is in charge of a new system of obtaining data. The labelled data this project started with was insufficient to train our initial model as well as desired, so we designed this system to use the model's predictions for iterative improvement. Since we have plenty of predictions on test images from our slide_testing/, this section presents these classifications to the user. As these classifications are presented to the user, the mistakes are corrected using an interactive gui. Once the user is finished classifying images in their interactive session, quitting the system saves all the newly obtained labelled data. 

  This new labelled data can then be used to train again, improving the results that are presented with each iteration of additional training data. Because of this iterative improvement, there were fewer mistakes to correct, resulting in quicker production of more labelled data.



At this point, you can use the final scripts to generate any sort of results/statistics automatically from the given slides in the data directory. I have yet to make these (sorry, the project isn't finished yet as I write this).

#### Thanks, and good luck, have fun!

