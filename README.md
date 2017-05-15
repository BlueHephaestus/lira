# Description

L.I.R.A. (Lesion Image Recognition and Analysis) is a machine learning for image recognition and analysis project made in collaboration with [Colorado State University's Mycobacteria Research Labs.](http://mrl.colostate.edu/)

This project was made to help identify the frequency of lesions in infected tuberculosis lung slides, however it has been made efficiently, modularly, and with good documentation, in the hopes that others may also find use in it's various tools. In having a quickly automated system for testing tuberculosis drug treatment, treatments can be tested faster and new statistics can be gained for the efficacy of these treatments.

The problem has been approached with a machine learning pipeline, consisting of deep convolutional neural networks in Python's Keras (v2.0) library, object detection with Python and OpenCV (v3.0), along with various other techniques in machine learning and neural networks. 

I plan to include the specific hyperparameters and configuration details for these techniques in the research paper to be published at the time of this project's completion, however if I do not I will put them here. Despite this, many specifics and details can be found in this README.

This project/repo will be made public along with a research paper, found [here when it's finished]()

The dataset used for this project is too large to store here, however we will set up a system for ease of access to others, if needed.

I, Blake Edwards (https://github.com/DarkElement75), am manager of the code for this project, as the primary developer of the code for the project and official collaborator with the MRL. Much of the initial development history for this project is not available, due to the fact that we didn't have Github Premium during that time, and thus couldn't make a private repository. This repository was then made private during the majority of the remainder of the project's development, until we published the research paper.

We hope that the code here has been documented and designed well enough that many people may modify it for their own uses, but if you have trouble with any of it, don't hesitate to contact the manager of the code for this project (Blake Edwards / Dark Element).

The project is divided into several directories, described below. Every program in these directories (with the exception of LIRA MK1) contains it's own detailed documentation, along with the functions in each of these programs. If anything is not clear enough, I would greatly appreciate any questions or helpful comments. 

## Current Machine Learning Pipeline

Note: This pipeline is subject to change, and the following description is only our current best strategy.

### Motivation for the pipeline architecture - Microscopic and Macroscopic classification

At the microscopic level (the level our Deep neural-network classifiers observe our data), our Type 1 and Type 3 lesions are nearly indistinguishable to the human eye. It seems they are also indistinguishable to the machine eye, as it's performance was very poor at differentiating between these two. 

When pathologists / microbiologists classify a Type 1 or Type 3, they can very easily tell the difference simply by looking at the slides from a macroscopic, or larger, scale. They zoom out the image, and it's obvious. So, we realized in order to fix this problem we'd need to give our network this ability, to zoom out and look to see which big sections are type 1, and which are not.

### Part 1 - Preparing a Type 1 Object Detector

Fortunately, object detection is a common machine learning problem. Given a big slide, since we are doing macroscopic detection (on the massive Type 1 lesion sections on these slides), we can lower the resolution in order to see it like a human would. Since we have lower resolution, it also means we don't have to have a huge input size for our detection system, which saves on memory and computational costs. Due to the average size of our hand-picked Type 1 samples being around 2048x2048, we decided to resize down by a factor of 4, using inputs of size 512x512.

In order to detect, we needed something trained for detection, and in order to do that, we needed training data. So we hand-picked some positive examples  of Type 1 lesions (and resized them down to 512x512), and then scanned a 2048x2048 window across some slides without any Type 1s in order to get a large amount of negative samples, which were then resized down to 512x512 again. 

In summary, the plan was to resize our test images down by a factor of 4, and then scan across them with a window of size 512x512, as our detector was trained on images which were originally 2048x2048, then resized to 512x512 - just like our test images would be.

For each of these 512x512 samples for training, we decided to use Histogram of Oriented Gradients (HOG for short) to extract features. We then trained an SVM on this new, more compact, and more representative dataset of features. To be clear, this SVM was trained to predict a positive when encountering a macroscopic Type 1 lesion, and a negative on anything else.

### Part 2 - Using our Type 1 Object Detector

Once we've trained our detector using the above method, we can use it in the following way: 

1. Given a new (full resolution) slide, we resize it down 4x. 

2. We then scan a window across our (low resolution) slide, using our previously trained SVM to detect any Type 1 lesions in our slide. We then scale our slide down further and scan again, repeating this process using OpenCV's detectMultiScale ([described here](http://www.pyimagesearch.com/2015/11/16/hog-detectmultiscale-parameters-explained/)) until complete.

3. We then remove overlapping predicted bounding rectangles using a mean-shift or non-maxima suppression algorithm, mean-shift being something built into OpenCV's detectMultiScale. 

4. At this point, we have obtained all bounding rectangles for Type 1 lesion classifications. In order to use this for our advantage however, we need to train two microscopic classifiers.

### Part 3 - Preparing two microscopic classifiers to use our bounding rectangles

Once we have bounding rectangles for our Type 1 lesions, we can simply use one microscopic classifier to get the individual classifications inside this rectangle, and another to get everything outside of it. First however, we have to train them.

1. Samples for training are obtained through various means. These could be obtained via LIRA-Live (described below), or just through a quick script to create an h5py archive from a directory of samples. The samples in this archive are 7 classes initially (shown at large-scale / zoomed out): ![Classifications](/documents/classifications.png)

2. The archive is split into two new archives: one containing the Healthy Tissue, Empty Slide, Type 1 - Rim, and Type 1 Caseum classifications, the other containing Healthy Tissue, Empty Slide, Type 2, and Type 3 classifications.

3. Our first classifier (Type 1 classifier) is trained on the first archive, and our second classifier (Type 2 & 3 classifier) is trained on the second archive. What this does is it prepares our first classifier for everything Type 1, and our second classifier for everything Type 2 or 3. They both come equipped with Healthy Tissue & Empty Slide classifications so that they can correctly classify any of those they encounter. 

For an idea of the microscopic model, it originally looked like this when training on all 7 classifications: ![Microscopic Model](/documents/model_graphic_1.png). This model is trained entirely, and is not a transfer learning model. For the two microscopic models, the model architecture is exactly the same as this visual, with the exception of the number of outputs, which is 4 instead of 7.
 
### Part 4 - Using our microscopic classifiers

As you may notice, the input size for the microscopic classifiers is 80x145, far smaller than the 65,000x30,000 (approximate) size of our full slides. We classify the full slides by dividing them into a lot of 80x145 subsections, iterating across these, and classifying each with the appropriate classifier. We classify them with our first classifier if they are inside of a bounding rectangle, and classify them with our second if they are not.

### Part 5 - Cleaning our classifications

Once we've finished this, the raw results should be far superior to those obtained before. However, there tend to be occasional small mistakes on such massive slides, as well as small mistakes near the borders of the bounding rectangles due to disagreements between the two microscopic classifiers.

Fortunately, we already have a Denoising algorithm for our predictions / labels across our slide, which we use to clean / smooth this up.

Below is a breakdown of each of the project directories, and their purpose(s).

## lira/

This directory contains the data, saved models, and files for training the Keras model (currently a deep convolutional neural network) for microscopic lesion classification.

### MK 1

  This version of LIRA was eased by use of separate bots - [DENNIS](https://github.com/DarkElement75/dennis) for the deep learning system it referenced in configuration, and [BBHO](https://github.com/DarkElement75/bbho) for Bayesian Optimization of that configuration. experiment_config.py is a configuration file for DENNIS MK 6, and lira_optimization_input_handler.py and lira_configurer.py are configuration files for use with BBHO.

  The remaining files are used with getting the data formatted so that DENNIS MK 6 can use it to train, and for creating archives that will be useful later. There are some older files and deprecated data handlers as well.

### MK 2 - LIRA Harder
  
  This version was made to be much more compact, readable, and completely independent of DENNIS. With DENNIS, we fell into [the generalization problem](https://xkcd.com/974/), where at the cost of efficiency/readability I made a system that had tons of options, and could handle more experiments. Unfortunately, this is pointless when I can now create a simple file (as I have now done) with Keras for any problem, with all the customisability of Keras (i.e. a lot).I've now done that with LIRA MK 2; it now minimally exists in a much simpler lira2.py file. It also has numerous other useful features detailed below, to help with model development and testing.

Features:
  1. Simpler, more compact, more modular, and more documented dataset handler for handling our training, validation, and test datasets.
  2. Numerous independent training iterations for random cross-validation, to increase confidence in true model performance.
  3. Saving of model results for future reference.
  4. Saving of model for future use.
  5. Saving of model metadata for future use.
  6. Simpler and more compact graphing of training loss, training accuracy, validation accuracy, and test accuracy across models, with an average result for each across models.

There have been many experiments and upgrades of the LIRA MK 2 model, denoted as LIRA MK 2.x . These experiments and the currently used version are detailed below:

**MK 2.0** - Initial version

**MK 2.1** - Improved hyperparameters and added denoising to latter portion of pipeline

**MK 2.2** - Testing with balancing of training data as well as augmentation of training data, all tests showed worse results than original data however not enough time was available to optimize the architecture for the modified dataset.

**MK 2.3** - Tested results on only 4 classifications (Healthy Tissue, Empty Slide, Type 2, and Type 3) instead of all 7 per usual. This improved results, but at the obvious cost of never classifying Type 1 lesions.

**MK 2.4** - Added new file to use transfer learning to train our microscopic classifier, however this had the same problems our original had - it was unable to correctly classify Type 1 classifications due to the lack of Type 1 data in the current datset.

**MK 2.5** - Added functionality to both transfer learning and non-transfer learning versions to support RGB data as well as gray data.

**MK 2.6** - Re-optimized both non-transfer learning and transfer learning versions with new dataset, balanced between all classifications. Results improved, however both versions still had trouble differentiating between Type 1 and 3 lesions.

**MK 2.7** - Updated transfer learning to fine-tune full model after training a small add-on / bottleneck model, however the problem still remained.

**MK 2.8** - Currently WIP, macroscopic + cooperative microscopic classification as described above. Expected to solve the Type 1 and Type 3 problem.

## lira_static/

  This section of the project is in charge of classifying our test slides, using our model(s) once they are finished training, as well as getting displayable results once the test slides are classified. All of the files here are devoted to loading the model(s), generating results, and storing and prettifying those results. Functions include:

1. **generate_predictions.py** - Main file for classifying test slides (i.e. generating predictions on those slides), using other files in lira_static/

2. **img_handler.py** - Contains many miscellaneous functions for use across the LIRA project.

3. **object_detection_handler.py** - In charge of loading and using our SVM for getting bounding rectangles on our slides.

4. **static_config.py** - In charge of loading and using our microscopic classifiers for individual subsection classification.

5. **post_processing.py** - Loads our predictions obtained from generate_predictions, and uses our denoising algorithm to clean up / smooth them.

6. **generate_display_results.py** - Loads our predictions obtained from generate_predictions and combines them with our grayscales into a colored overlay, then resizes them down so they can be easily opened and viewed.

7. **concatenate_results.py** - Since our generate_display_results can only do it's thing on subsections of the original slides (due to memory constraints), we concatenate all of those images together with this file.
 

## lira_live/

  This section of the project is in charge of a new system of obtaining data. The labelled data this project started with was insufficient to train our initial model as well as desired, so we designed this system to use the model's predictions for iterative improvement. Since we have plenty of predictions on test images from our lira_static/ (presented without denoising applied), this section presents these classifications to the user. As these classifications are presented to the user, the mistakes are corrected using an interactive gui. Once the user is finished classifying images in their interactive session, quitting the system saves all the newly obtained labelled data. 

  This new labelled data can then be used to train again, improving the results that are presented with each iteration of additional training data. Because of this iterative improvement, there were fewer mistakes to correct, resulting in quicker production of more labelled data.

## Object Detection tests

We currently have been experimenting and training our macroscopic object detector on a separate project file outside of this one, however it will soon be moved to an appropriate place in this repository once ready.

## Final Notes 

At this point, you can use the final scripts to generate any sort of results/statistics automatically from the given slides in the data directory. I have yet to make these (sorry, the project isn't finished yet as I write this).

### Thanks, and good luck, have fun!

