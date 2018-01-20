"""
Our "pipeline" file to go through the 7 major stages of LIRA which make up our LIRA pipeline.
A summary of what each of our 7 stages does and their associated directory/files are as follows:
    1. Raw Images -> Image archive (get_archive.py)
    2&3. 
        Sample archive + Type 1 Detection (Macroscopic) Model -> Detected Type 1 Lesions for Images / Samples (lira_static/generate_detected_objects.py)
        Detected Type 1 Lesions -> Suppressed Detected Type 1 Lesions (lira_static/generate_detected_objects.py)
    4. Suppressed Detected Type 1 Lesions -> Human-double-checked Detected Type 1 Lesions (lira_object_detection_editing/object_detection_editor.py)
    5. Image archive + Microscopic Model + Human-double-checked Detected Type 1 Lesions  -> Predictions Archive (lira_static/generate_predictions.py)
    6. Image Archive + Predictions Archive -> Human-double-checked Predictions Archive (lira_microscopic_prediction_editing/microscopic_prediction_editor.py)
    7. Image Archive + Human-double-checked Predictions Archive -> Jpg predictions per image (lira_static/generate_display_results.py)

We have to set the directories again in the arguments because of where this file is positioned. When we have "../" something in one of our files for the directory, it is relative to the parent file calling that function. When that function is here, i.e. not in the normal place, it messes up stuff sometimes.

So use this when you want to run through the entire LIRA pipeline;
    Use wisely, may take days to complete and could cause problems if interrupted. 
    Uses a lot of processing power for nearly all stages.
    You have been warned.
Good luck, have fun!

-Blake Edwards / Dark Element
"""

"""
TEMPORARY - WHILE MEASURING RESULTS WITH EXPERIMENTS
"""
import shutil, h5py, pickle
import numpy as np

import sys, os

sys.path.append("lira/lira2/src")

import get_archive

sys.path.append("lira_object_detection_editing")

import object_detection_editor

sys.path.append("lira_static")

import generate_detected_objects
import generate_predictions
import generate_display_results

sys.path.append("lira_microscopic_prediction_editing")

import microscopic_prediction_editor

def main(model_title, 
            detection_model_title, 
            img_archive_dir="lira/lira1/data/images.h5",
            resized_img_archive_dir="lira/lira1/data/resized_images.h5",
            predictions_archive_dir="lira/lira1/data/test_predictions.h5",
            rects_archive_dir="lira/lira1/data/bounding_rects.pkl",
            classification_metadata_dir="lira_static/classification_metadata.pkl"):
    """
    Arguments:
        model_title: Name to use for our two microscopic models.
        detection_model_title: Name to use for our macroscopic model.
        img_archive_dir: Filepath of our images. 
        resized_img_archive_dir: a string filepath of the .h5 file where the resized images will be stored.
        predictions_archive_dir: Filepath of our predictions.
        rects_archive_dir: a string filepath of the .pkl file where the detected rects will be stored.
        classification_metadata_dir: Filepath of our classification metadata. Will be used for handling our predictions / classifications properly.

    Returns:
        As stated in the header documentation for this file, we go through the 7 major stages of our pipeline (2 of which are in the same file) with our 6 program calls below.
        If you wish for further details on each stage, they are in the research paper, in the github repository, and in the independent files for each.
    """

    """
    Get filename of nns from title
    """
    model = model_title.lower().replace(" ", "_")
    model1 = model + "_model_1"
    model2 = model + "_model_2"
    detection_model = detection_model_title.lower().replace(" ", "_")
    
    """
    TEMPORARY - WHILE MEASURING RESULTS WITH EXPERIMENTS
    """
    user_id = raw_input("Input your user id, if you do not have one choose one to use throughout sessions: ")
    resize_factor = float(raw_input("Input the resize factor as a decimal number like 0.34, between 0.0 and 1.0: "))
    restart_session = raw_input("Input 'y' (without quotes) if you want to start your session over again. Input anything else to continue: ")

    #Get our base and user predictions archive dir, so we can handle both cases where we are and are not classifying from scratch.
    #base_predictions_archive_dir="lira/lira1/data/test_predictions.h5"
    predictions_archive_dir="lira/lira1/data/%s_test_predictions.h5" % user_id

    interactive_session_metadata_dir="lira_microscopic_prediction_editing/%s_interactive_session_metadata.pkl" % user_id
    rects_archive_dir="lira/lira1/data/%s_bounding_rects.pkl" % user_id

    regenerating_rects = False
    regenerating_predictions = False
    if not os.path.isfile(interactive_session_metadata_dir):
        #Create it if it doesn't exist
        open(interactive_session_metadata_dir,"w").close()
        regenerating_predictions = True

        with open(interactive_session_metadata_dir, "r+") as f:
            pickle.dump((-1,0.15),f)

    if not os.path.isfile(rects_archive_dir):
        regenerating_rects = True
        open(rects_archive_dir,"w").close()

    if not os.path.isfile(predictions_archive_dir):
        regenerating_predictions = True
        open(predictions_archive_dir, "w").close()

    if restart_session.lower() == 'y':
        #Reset it if they want
        regenerating_rects = True
        regenerating_predictions = True
        open(predictions_archive_dir, "w").close()
        open(rects_archive_dir,"w").close()
        with open(interactive_session_metadata_dir, "r+") as f:
            pickle.dump((-1,0.15),f)

    classifying_from_scratch = False

    """
    From our test_slides dir, generate new images.h5 file for later
        (If we want to, and there isn't a massive amount of images we want to avoid)
    """
    print "Getting Images Archive..."
    #get_archive.create_archive("lira/lira1/data/rim_test_slides", "lira/lira1/data/test.h5", rgb=True)
    #get_archive.create_archive("/home/basay3/Desktop/lira_validation/Week1", "lira/lira1/data/experimental_images.h5", rgb=True)

    """
    Get our detected objects/rects on our images, using our object detection model.
    """
    print "Generating Bounding Rects..."

    """
    Create our necessary .h5 files since there isn't a write option I can specify to do so with h5py
    """

    if regenerating_rects:
        generate_detected_objects.generate_detected_objects(detection_model, model_dir="lira/lira2/saved_networks", img_archive_dir=img_archive_dir, rects_archive_dir=rects_archive_dir)

    """
    Present detected objects/rects to user, so that they can correct any mistakes using our UI tool.
    """
    print "Displaying Macroscopic Predictions / Detected Objects for Human-in-the-loop classification..."

    """
    Create our necessary .h5 files since there isn't a write option I can specify to do so with h5py """ #open(resized_img_archive_dir,"w").close()

    if regenerating_rects:
        object_detection_editor.edit_detected_objects(rect_h=640, rect_w=640, step_h=320, step_w=320, resize_factor=0.1, img_archive_dir=img_archive_dir, resized_img_archive_dir=resized_img_archive_dir, rects_archive_dir=rects_archive_dir, dual_monitor=False, images_already_resized=False)

    """
    From our saved model and images, generate new predictions.h5 file
    """
    print "Generating Predictions..."

    """
    Create our necessary .h5 files since there isn't a write option I can specify to do so with h5py
    """
    #open(predictions_archive_dir,"w").close()
    if regenerating_predictions:
        generate_predictions.generate_predictions(model1, model2, model_dir = "lira/lira2/saved_networks", img_archive_dir = img_archive_dir, predictions_archive_dir = predictions_archive_dir, rects_archive_dir = rects_archive_dir, classification_metadata_dir = "lira_static/classification_metadata.pkl", rgb=True)
    
    """
    TEMPORARY - WHILE MEASURING RESULTS WITH EXPERIMENTS
    Create a bunch of blank predictions if we are classifying from scratch
        by looping through the images creating an archive and saving new predictions for each as entirely empty slide
    Create these ONLY IF the user has not done this before, i.e. our prev_img_i < 0
    """
    #f = open("lira_microscopic_prediction_editing/interactive_session_metadata.pkl", "r")
    f = open(interactive_session_metadata_dir, "r")
    prev_img_i, alpha = pickle.load(f)
    f.close()
    if (classifying_from_scratch and prev_img_i < 0):
        #This will create the file if we haven't started the session for this user and they are classifying from scratch.
        #If they have started the session and are classifying from scratch, we do nothing.
        with h5py.File(img_archive_dir, "r", chunks=True, compression="gzip") as img_hf:
            with h5py.File(predictions_archive_dir, "w") as predictions_hf:
                img_n = len(img_hf.keys())
                for img_i in range(img_n):
                    img_shape = img_hf.get(str(img_i)).shape
                    predictions = np.zeros((img_shape[0]//80, img_shape[1]//145, 7),int)
                    predictions[:,:,3] = 1#all 3 one-hot vectors
                    predictions_hf.create_dataset(str(img_i), data=predictions)

    """
    I don't think we need this. If they're not classifying from scratch, they're doing cooperative classification, and if they haven't started yet,
        then they should already have predictions because they've been generated.

    If they have started, then of course they already have them.
    elif (not classifying_from_scratch and prev_img_i < 0):
        #If they aren't classifying from scratch and the session hasn't started for this user, we copy our base predictions to be the user's predictions
        shutil.copy(base_predictions_archive_dir, predictions_archive_dir)
    """
    


    """
    TEMPORARY - WHILE MEASURING RESULTS WITH EXPERIMENTS
    Copy our predictions to a pre_editing state
    """
    shutil.copy(predictions_archive_dir, predictions_archive_dir + "_pre_editing")

    """
    From our predictions and images, we allow the user to look at them once more before moving on to get displayable results, statistics, and so on.
    """
    print "Displaying Microscopic Predictions for Human-in-the-loop classification..."
    microscopic_prediction_editor.edit_microscopic_predictions(img_archive_dir = img_archive_dir, resized_img_archive_dir = resized_img_archive_dir, predictions_archive_dir = predictions_archive_dir, classification_metadata_dir=classification_metadata_dir, interactive_session_metadata_dir=interactive_session_metadata_dir, dual_monitor=False, images_already_resized=False, resize_factor=resize_factor)

    """
    TEMPORARY - WHILE MEASURING RESULTS WITH EXPERIMENTS
    Get stats between our prediction files
        save as two matrices of shape image_n x class_n x 2
    """
    #Get proportions for pre_predictions
    if classifying_from_scratch:
        with h5py.File(predictions_archive_dir, "r") as post_predictions_hf:
            img_n = len(post_predictions_hf.keys())
            class_n = 7
            post_stats = np.zeros((img_n, class_n, 2))

            for img_i in range(img_n):
                post_predictions = np.argmax(np.array(post_predictions_hf.get(str(img_i))), axis=2)
                prediction_n = float(post_predictions.size)
                for class_i in range(class_n):
                    post_stats[img_i, class_i, 0] = np.sum(post_predictions==class_i)
                    post_stats[img_i, class_i, 1] = post_stats[img_i, class_i, 0] / prediction_n
        #We're done, save our stats
        with open("%s_results.pkl"%user_id, "w") as f:
            pickle.dump(post_stats, f)
    else:
        with h5py.File(predictions_archive_dir+"_pre_editing", "r") as pre_predictions_hf:
            with h5py.File(predictions_archive_dir, "r") as post_predictions_hf:
                img_n = len(pre_predictions_hf.keys())
                class_n = 7
                pre_stats = np.zeros((img_n, class_n, 2))
                post_stats = np.zeros((img_n, class_n, 2))

                for img_i in range(img_n):
                    pre_predictions = np.argmax(np.array(pre_predictions_hf.get(str(img_i))), axis=2)
                    post_predictions = np.argmax(np.array(post_predictions_hf.get(str(img_i))), axis=2)
                    prediction_n = float(pre_predictions.size)
                    for class_i in range(class_n):
                        pre_stats[img_i, class_i, 0] = np.sum(pre_predictions==class_i)
                        pre_stats[img_i, class_i, 1] = pre_stats[img_i, class_i, 0] / prediction_n

                        post_stats[img_i, class_i, 0] = np.sum(post_predictions==class_i)
                        post_stats[img_i, class_i, 1] = post_stats[img_i, class_i, 0] / prediction_n

        #We're done, save our stats
        with open("%s_results.pkl"%user_id, "w") as f:
            pickle.dump((pre_stats, post_stats), f)

    """
    From our new predictions.h5 and greyscales, generate easily accessible images for viewing.
    """
    print "Generating Display Results..."
    #generate_display_results.generate_display_results(img_archive_dir = img_archive_dir, predictions_archive_dir = predictions_archive_dir, classification_metadata_dir = "lira_static/classification_metadata.pkl", results_dir = "lira_static/results", alpha=0.33, neighbor_weight=0.8, rgb=True)

    """
    At this point we have all we need to open LIRA live again when we want to, so we are done!
    """
    print "Completed! -DE"

main("LIRA MK2.8 Cooperative Model Classification", "Type1 Detection Model MK5", img_archive_dir="lira/lira1/data/experimental_images.h5", resized_img_archive_dir="lira/lira1/data/resized_images.h5", predictions_archive_dir="lira/lira1/data/test_predictions.h5", rects_archive_dir="lira/lira1/data/bounding_rects.pkl",  classification_metadata_dir="lira_static/classification_metadata.pkl")
