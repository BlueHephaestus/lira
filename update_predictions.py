"""
Meta file to make use of the major stages of LIRA in order to do:
    1. Raw Images/Samples -> Sample archive (get_archive.py)
    2. Sample archive -> New trained model (lira2.py / lira2_pre_transfer_learning.py)
    3. Trained model & Greyscale archive -> Predictions Archive (generate_predictions.py)
    4. Predictions Archive & Greyscale Archive -> Jpg predictions per image (generate_display_results.py)

We have to set the directories again in the arguments because of where this file is positioned. When we have "../" something in one of our files for the directory, it is relative to the parent file calling that function. When that function is here, i.e. not in the normal place, it messes up stuff sometimes.

So use this when you want to update the end predictions, usually will be used once more data is available for the model.
    Use wisely, may take days to complete and could cause problems if interrupted. 
    Uses a lot of processing power for nearly all stages.
    You have been warned.
Good luck, have fun!

-Blake Edwards / Dark Element
"""

import sys

sys.path.append("lira/lira2/src")

import get_archive
import lira2_pre_transfer_learning_mk2

sys.path.append("lira_static")

import generate_predictions
import generate_display_results

def main(model_title, detection_model_title, img_archive_dir, predictions_archive_dir):
    """
    Get filename of nn from title
    """
    model = model_title.lower().replace(" ", "_")
    detection_model = detection_model_title.lower().replace(" ", "_")
    
    """
    From our test_slides dir, generate new images.h5 file for later
        (If we want to, and there isn't a massive amount of images we want to avoid)
    """
    print "Getting Images Archive..."
    #get_archive.create_archive("lira/lira1/data/rim_test_slides", "lira/lira1/data/rim_test_images.h5", rgb=True)

    """
    From our samples.h5 file, train our model and save in saved_networks/`nn`
        You need to change your input dimensions in the model training file, if you want to switch between grayscale/rgb
    """
    print "Training Models..."
    """
    Train our first model on our samples for this model
    """
    model1 = model + "_model_1"
    lira2_pre_transfer_learning_mk2.train_model(model1, model_dir="lira/lira2/saved_networks", archive_dir="lira/lira2/data/augmented_model_1_samples.h5")

    """
    Train our second model on our samples for this model
    """
    model2 = model + "_model_2"
    lira2_pre_transfer_learning_mk2.train_model(model2, model_dir="lira/lira2/saved_networks", archive_dir="lira/lira2/data/augmented_model_2_samples.h5")

    """
    From our saved model and greyscales, generate new predictions.h5 file
    """
    print "Generating Predictions..."
    generate_predictions.generate_predictions(model1, model2, detection_model, model_dir = "lira/lira2/saved_networks", img_archive_dir = img_archive_dir, predictions_archive_dir = predictions_archive_dir, classification_metadata_dir = "lira_static/classification_metadata.pkl", rgb=True)

    """
    From our new predictions.h5 and greyscales, generate easily accessible images for viewing.
    """
    print "Generating Display Results..."
    generate_display_results.generate_display_results(img_archive_dir = img_archive_dir, predictions_archive_dir = predictions_archive_dir, classification_metadata_dir = "lira_static/classification_metadata.pkl", results_dir = "lira_static/results", alpha=0.33, neighbor_weight=0.8, rgb=True)
#
    """
    At this point we have all we need to open LIRA live again when we want to, so we are done!
    """
    print "Completed! -DE"

main("LIRA MK2.8.1 Upgraded Cooperative Model Classification", "Type1 Detection Model MK5", "lira/lira1/data/images.h5", "lira/lira1/data/test_predictions.h5")
