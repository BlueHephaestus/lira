"""
Meta file to make use of the major stages of LIRA in order to do:
    1. Raw samples -> Sample archive (get_archive.py)
    2. Raw Greyscales -> Greyscale archive (get_greyscales.py)
    3. Sample archive -> New trained model (experiment_config.py)
    4. Trained model & Greyscale archive -> Predictions (generate_overlay.py)
    5. Predictions -> Jpg predictions per image (concatenate_results.py)

We have to set the directories again in the arguments because of where this file is positioned. When we have "../" something in one of our files for the directory, it is relative to the parent file calling that function. When that function is here, i.e. not in the normal place, it messes up stuff sometimes.

So use this when you want to update the end predictions, usually will be used once more data is available for the model.
    Use wisely, may take days to complete and could cause problems if interrupted. 
    Uses a lot of processing power for nearly all stages.
    You have been warned.
Good luck, have fun!

-Blake Edwards / Dark Element
"""
"""
OI FUTURE SELF
    Alright so our overlays are entirely pink, 
    and our accuracies are entirely too perfect.
    I think I see the problem here.
    We aren't importing our data into a dataset object correctly when using live_samples.h5 (and no other samples)
    It seems to be just storing them all as one label.
    Get them to use the metadata we have if you can, and fixing that should fix the rest of it.
    Good luck, have fun!
"""

import sys

sys.path.append("lira/lira1/src")

import get_archive
import experiment_config

sys.path.append("slide_testing")

import generate_overlay
import concatenate_results

def main(nn_title):
    """
    Get filename of nn from title
    """
    nn = nn_title.lower().replace(" ", "_")
    
    """
    From our sample subdirectories and live_samples.h5 files, generate a samples.h5 file
    """
    print "Getting Samples Archive..."
    #get_archive.regenerate_archive(archive_dir="lira/lira1/data/samples.h5", data_dir="lira/lira1/data/samples")

    """
    From our test_slides dir, generate new greyscales.h5 file for later
        (If we want to, and it isn't a massive amount of greyscales we want to avoid)
    """
    print "Getting Greyscales Archive..."
    #get_greyscales.load_greyscales("../data/test_slides", "../data/greyscales.h5")

    """
    From our samples.h5 file, train our model and save in saved_networks/`nn`
    """
    print "Training Model..."
    #experiment_config.train_model(nn, nn_dir="lira/lira1/saved_networks")

    """
    From our saved model and greyscales, generate new predictions.h5 file
    """
    print "Generating Predictions..."
    #generate_overlay.generate_predictions(nn)
    generate_overlay.generate_predictions(nn, nn_dir = "lira/lira1/src", img_archive_dir = "lira/lira1/data/greyscales.h5", predictions_archive_dir = "lira/lira1/data/predictions.h5", classification_metadata_dir = "slide_testing/classification_metadata.pkl", results_dir = "slide_testing/results")

    """
    Concatenate results of generating overlay, for showing off and debugging if we want to
    """
    print "Concatenating Results..."
    concatenate_results.concatenate_results("slide_testing/results/", "slide_testing/concatenated_results/", classification_metadata_dir="slide_testing/classification_metadata.pkl")
    
    """
    At this point we have all we need to open LIRA live again when we want to, so we are done!
    """
    print "Completed! -DE"

main("New LIRA Live Data MK1_1")
