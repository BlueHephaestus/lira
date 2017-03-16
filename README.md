## Description

L.I.R.A. (Lesion Image Recognition and Analysis) is a machine learning for image recognition and analysis project made in collaboration with [Colorado State University's Microbiology Research Lab](http://mrl.colostate.edu/), with the vast majority of the artificial intelligence portion of the work done by myself, Blake Edwards / Dark Element.

In summary, this project was made to help identify the frequency of lesions in infected tuberculosis lung slides. In having a quickly automated system, treatments can be tested faster and more information can be gleaned with better precision than before. The problem has been approached with a deep convolutional neural network in Python's Keras library. The remaining information and parameters can be found in the research paper.

This project has been made public along with a research paper, found [here when it's finished]()

I am manager of the code for this project, as the primary developer of the code for the project and official collaborator with the MRL. Unfortunately, updating this as it was developed publicly is not an option before the paper was published, so much of this was developed and replaced on my own machines. Because of that, much of the history of the development of this project is not available, and a large amount of it is merely marked under it's own "Initial Commit". Github Premium was purchased at this point, so that I could continue updating the remaining history as it was finished, and then make the repository public when the time came.

The dataset is too large to store here, however we will set up a system for ease of access to others.

The project is divided into several directories, described below.

#### lira/

##### MK 1

  The main section of the project, this directory contains the data for training the Keras deep convolutional neural network configured inside. Much of this development was eased by use of separate bots [DENNIS](https://github.com/DarkElement75/dennis) for the deep learning system LIRA references in configuration, and [BBHO](https://github.com/DarkElement75/bbho) for Bayesian Optimization of that configuration. experiment_config.py is a configuration file for DENNIS MK 6, and lira_optimization_input_handler.py and lira_configurer.py are configuration files for use with BBHO.

  The remaining files are used with getting the data formatted so that DENNIS MK 6 can use it to train, and for creating archives that will be useful later. There are some older files and deprecated data handlers as well.


#### slide_testing/

This section of the project is in charge of classifying our test slides, using our network once it is finished training. All of the files here are devoted to loading the network, generating results, and storing and prettifying those results.

#### slide_labeller/

This section of the project is in charge of a new system of obtaining data. The labelled data we started with was insufficient to train a neural network well enough, so I designed this system to use what we had to make more data. There are plenty of test slides that the network can be run on to quickly generate results, so this section takes those new classifications and gives users GUI tools to correct the classifications until the entire presented area is classified. Once the user is done classifying, they can quit the system and it will save all the new data. Since the network can easily use this new data to train again, it improves its own results every time, making fewer mistakes to correct. Since it makes fewer mistakes, the time to correct them also decreases, and more data is produced with every time this process is repeated. 

This process can be repeated as many times as desired, or until there is no more unlabelled data. Ideally, the network will improve accuracy until reaching above human level accuracy (or perhaps I should say microbiology graduate student accuracy). 

At this point, you can use the final scripts to generate any sort of results/statistics automatically from the given slides in the data directory. I have yet to make these (sorry, the project isn't finished yet as I write this).

#### Thanks, and good luck, have fun!

