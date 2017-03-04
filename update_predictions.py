"""
Meta file to make use of the major stages of LIRA in order to do:
    1. Raw samples -> Sample archive (get_archive.py)
    2. Raw Greyscales -> Greyscale archive (get_greyscales.py)
    3. Sample archive -> New trained model (experiment_config.py)
    4. Trained model & Greyscale archive -> Predictions (generate_overlay.py)
    5. Predictions -> Jpg predictions per image (concatenate_results.py)

So use this when you want to update the end predictions, usually will be used once more data is available for the model.
    Use wisely, may take days to complete and could cause problems if interrupted. 
    Uses a lot of processing power for nearly all stages.
    You have been warned.
Good luck, have fun!

-Blake Edwards / Dark Element
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
    get_archive.regenerate_archive()

    """
    From our test_slides dir, generate new greyscales.h5 file for later
    """
    get_greyscales.load_greyscales("../data/test_slides", "../data/greyscales.h5")

    """
    From our samples.h5 file, train our model and save in saved_networks/`nn`
    """
    experiment_config.train_model(nn)

    """
    From our saved model and greyscales, generate new predictions.h5 file
    """
    generate_overlay.generate_predictions(nn)

    """
    Concatenate results of generating overlay, for showing off and debugging if we want to
    """
    concatenate_results.concatenate_results("results/", "concatenated_results/", classification_metadata_dir="classification_metadata.pkl")
    
    """
    At this point we have all we need to open LIRA live again when we want to, so we are done!
    """

main("New LIRA Live Data MK1.1")
