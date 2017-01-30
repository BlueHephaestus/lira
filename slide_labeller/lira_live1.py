"""
This file is our main file for LIRA-Live, 
    where all of our child classes/functions get referenced and used for the entire process 
    of loading, displaying, updating, saving, and retraining.

-Blake Edwards / Dark Element
"""

import sys, h5py, pickle
import numpy as np

import PIL
from PIL import Image

sys.path.append("../slide_testing/")

import img_handler
from img_handler import *

import subsection_handler
from subsection_handler import *

import interactive_gui_handler
from interactive_gui_handler import *

def main(sub_h=80, 
         sub_w=145, 
         img_archive_dir="../lira/lira1/data/greyscales.h5",
         predictions_archive_dir="../lira/lira1/data/predictions.h5",
         classification_metadata_dir="../slide_testing/classification_metadata.pkl",
         interactive_session_metadata_dir="interactive_session_metadata.pkl",
         live_archive_dir="../lira/lira1/data/samples/live_samples.h5"):
    """
    Our main execution for LIRA-Live.

    After opening our archives,
    1. We loop through each image index, with which we can get the entire image and the assorted predictions.
    2. As we loop through our images, we don't yet know the number of subsections (sub_n) each has. 
    3. However, we can use get_relative_factor to get our division factor, since it is the same method we used
        when we generated the predictions and overlay. 
    4. Once we do that, we can loop through the number of subsections we should have, so we only go to the next image in the archive once we are done with the current one.
    5. As we loop through with our img_i and sub_i, we can then get our next overlay subsection with subsection_handler.py 's get_next_overlay_subsection(), 
        which we pass all the necessary values
    """

    """
    Load our classification metadata
    """
    f = open(classification_metadata_dir, "r")
    classifications, colors = pickle.load(f)
    f.close()

    """
    Get our metadata from any last sessions, 
        then keep the metadata directory open for any updates to interactive ui parameters
    """
    f = open(interactive_session_metadata_dir, "r")
    prev_img_i, prev_sub_i, alpha = pickle.load(f)
    f.close()

    """
    We initialize some counter values to 0, we will increment them during our session.
    They keep track of 
        1. the number of individual subsections (sub_h, sub_w) classified, 
            only updated once we go to a next image
        2. the number of empty classifications classified, 
            also only updated once we go to a next image
    """
    total_classification_n = 0
    empty_classification_n = 0

    """
    Open all our archives,
        our image archive, and predictions archive(as r & w), respectively.
    """
    with h5py.File(img_archive_dir, "r") as img_hf:
        with h5py.File(predictions_archive_dir, "r+") as predictions_hf:

            """
            Get the number of images to loop through
            """
            img_n = len(img_hf.keys())
            
            """
            Open a new interactive session before getting our images,
                and loop from where we left off last to our img_n
            """
            interactive_session = InteractiveGUI(classifications, colors, sub_h, sub_w, alpha)
            for img_i in range(prev_img_i, img_n):
                """
                Get our image, and associated predictions
                """
                img = img_hf.get(str(img_i))
                img_predictions = predictions_hf.get(str(img_i))

                """
                Get our number of subsections for this image from our divide factor ^ 2,
                    then start looping through from where we left off last to our sub_n.
                """
                factor = get_relative_factor(img.shape[0], None)
                sub_n = factor**2
                for sub_i in range(prev_sub_i, sub_n):
                    """
                    Now that we have an img_i and sub_i for each iteration, 
                        We check if this subsection is entirely empty, according to if all the predictions are empty. 
                        This is based on the computation-saving assumption that if all our predictions for a subsection are empty, then we have an empty subsection.
                    """
                    prediction_sub = get_prediction_subsection(sub_i, factor, img_predictions)
                    if not all_predictions_empty(prediction_sub, classifications):
                        """
                        If it is not empty, we continue!
                        Get our img for this subsection to display from get_next_overlay_subsection()
                        As well as our predictions for this subsection using get_prediction_subsection()
                        """
                        overlay_sub = get_next_overlay_subsection(img_i, sub_i, factor, img, img_predictions, classifications, colors, alpha=alpha, sub_h=80, sub_w=145)

                        """
                        Set our interactive UI's predictions and image before starting the session and our main ui loop
                        """
                        interactive_session.np_img = overlay_sub
                        interactive_session.predictions = prediction_sub
                        interactive_session.start_interactive_session()

                        while True:
                            """
                            Main loop where we wait for flags from Interactive UI before doing anything else
                            """
                            if interactive_session.flag_refresh:
                                """
                                If our flag is set to refresh our overlay_sub with updated parameters, we:
                                    1. Get any updated values before deleting our previous session
                                """
                                alpha = interactive_session.alpha
                                prediction_sub = interactive_session.predictions

                                """
                                    2. Update main img_predictions matrix with updated predictions subsection
                                    3. Get our new overlay subsection overlay_sub using updated img_predictions
                                """
                                img_predictions = update_prediction_subsection(sub_i, factor, img_predictions, interactive_session.predictions)
                                overlay_sub = get_next_overlay_subsection(img_i, sub_i, factor, img, img_predictions, classifications, colors, alpha=alpha, sub_h=80, sub_w=145)

                                """
                                4. Reload the interactive session with new parameters
                                """
                                interactive_session = InteractiveGUI(classifications, colors, sub_h, sub_w, alpha)
                                interactive_session.np_img = overlay_sub
                                interactive_session.predictions = prediction_sub

                                """
                                5. Reset flag to false
                                6. Start the interactive session
                                """
                                interactive_session.flag_refresh=False
                                interactive_session.start_interactive_session()

                            if interactive_session.flag_next:
                                """
                                if our flag is set to go to the next image in our interactive session, 
                                    1. Get any updated values before deleting our previous session
                                """
                                alpha = interactive_session.alpha
                                prediction_sub = interactive_session.predictions
                                
                                """
                                2. Update main img_predictions matrix with updated predictions subsection
                                """
                                interactive_session.flag_next=False
                                break

                            if interactive_session.flag_quit:
                                """
                                if our flag is set to quit and end our interactive session,
                                We can begin exiting everything.
                                    (no need to mess with flags, all is kill)
                                We do get any updated values before deleting our previous session, however.
                                """
                                alpha = interactive_session.alpha
                                prediction_sub = interactive_session.predictions
                                break


                    """
                    Now we can continue onwards to checking if we selected quit.
                    """
                    if interactive_session.flag_quit:
                        """
                        If we are going to the next image, we will let our loop do so. 
                        However, if we are going to quit the session, we need to exit our loops, so we break again.
                        """
                        break
                    
                    """
                    Now that we are done with classifying this subsection (or it was skipped because it was entirely empty),
                        we increment our total_classification_n and empty_classification_n accordingly

                    We know that if we quit on this subsection, this code will not be executed. 
                    It will only be executed if the user has selected to go to the next one, meaning they are done.

                    1. Get number of empty classifications via same method to check if they were all empty.
                    """
                    empty_i = list_find(classifications, "Empty Slide")
                    empty_classification_n += np.sum(prediction_sub==empty_i)

                    """
                    2. Get number of total classifications via the raw number of items in our prediction_sub, now that we are finished with it.
                    """
                    total_classification_n += prediction_sub.size

                if interactive_session.flag_quit:
                    """
                    Same as above. We need to exit all of our loops if we are done with our session.
                    """
                    break
            """
            At this point, we have either gone through all of our images or quit our interactive session.
            So, we want to update any changes we made on the last image we updated:
            1. Get any updated values before leaving the session
            """
            alpha = interactive_session.alpha
            prediction_sub = interactive_session.predictions

            """
            2. Update main img_predictions matrix with updated predictions subsection
            """
            img_predictions = update_prediction_subsection(sub_i, factor, img_predictions, prediction_sub)

            """
            3. Delete our interactive session to save memory, we will need as much memory as we can get for the next part.
            """
            del interactive_session

            """
            Then we can continue to updating metadata:

            Now, (since we already updated our copies on disk of img_predictions) we update any metadata for this session.
            """
            print "Session ended. Updating and storing updated metadata..."
            f = open(interactive_session_metadata_dir, "w")
            pickle.dump((img_i, sub_i, alpha), f)
            f.close()

            """
            And now, all that is left is to save all of our updated samples for use in the next training session.
            In order to do this, we have to create two new np arrays to store them in, and also store them in the following format:
                X dataset shape = (sample_n, sub_h*sub_w) -> the flattened images
                Y dataset shape = (sample_n, ) -> the integer classification indices for each image.
            This is rather complex. So let's take it step by step.

            Modify our indices so that we only reference the images we have updated (the images behind the last modified one), 
                unless we have run out of images (img_i == img_n-1 && sub_i == sub_n-1), 
                in which case we get everything, including our most recently modified image.
            """
            print "Updating live_samples.h5 archive for training..."
            if img_i != img_n-1 or sub_i != sub_n-1:
                sub_i -= 1

            """
            Add 1 to our final end indices, so we can use them inclusively in loop splicing.
            """
            updated_img_n = img_i+1
            updated_sub_n = sub_i+1

            """
            Since we may not be including all of our empty classifications, which in part make up the total number of classifications, 
            we need to compute how many empty classifications we actually add.
            Once we've done that, we add it to the difference of the original total number of classifications and the original empty number of classifications,
                to get our updated total number of classifications.
            """
            updated_empty_classification_n = get_extra_empty_samples(total_classification_n, empty_classification_n, classifications)
            updated_total_classification_n = total_classification_n - empty_classification_n + updated_empty_classification_n

            """
            Initialize our final x and y arrays according to our updated_total_classification_n (the total number of individual subsection classifications), sub_h, and sub_w
            """
            x = np.zeros((updated_total_classification_n, sub_h*sub_w))
            y = np.zeros((updated_total_classification_n,))

            """
            Loop through img, large subsection, and then individual subsection. We go all the way to individual to ensure we don't add any more empty classifications than needed.
                Since our sub_i is only relevant for the last img we run through, we have to loop in a rather peculiar fashion.
            """
            """
            We initialize a counter for empty classifications stored, so we know when to stop storing them.
            """
            empty_classification_i = 0

            """
            We also initialize a counter for total classifications stored, 
                so we can insert them into our x and y arrays correctly as we go along.
            """
            total_classification_i = 0
            for img_i in range(updated_img_n):
                img = img_hf.get(str(img_i))
                img_predictions = predictions_hf.get(str(img_i))
                factor = get_relative_factor(img.shape[0], None)
                if img_i != updated_img_n-1:
                    """
                    If this is not the last image, we loop through all the subsections.
                    """
                    sub_n = factor**2

                else:
                    """
                    If this is the last image, we loop through only until our last subsection we encountered during the interactive session.
                    """
                    sub_n = updated_sub_n

                for sub_i in range(sub_n):
                    """
                    Now we know this loop will only loop through classified subsections, so we can just start looping through individual subsections.
                        (after we get the subsections)
                    """
                    row_i = sub_i//factor
                    col_i = sub_i % factor

                    greyscale_sub = get_next_subsection(row_i, col_i, img.shape[0], img.shape[1], sub_h, sub_w, img, factor)
                    prediction_sub = get_prediction_subsection(sub_i, factor, img_predictions)

                    """
                    Again, we can easily get the number of predictions in our subsection to get the number
                        of total predictions and subsequently individual subsections we have in this subsection.
                    """
                    individual_sub_n = prediction_sub.size

                    """
                    We then convert greyscale_sub and prediction_sub to their final format so we can easily loop through and reference them.
                    """
                    greyscale_sub = np.reshape(greyscale_sub, (individual_sub_n, sub_h*sub_w))
                    prediction_sub = prediction_sub.flatten()

                    for individual_sub_i in range(individual_sub_n):
                        """
                        And start looping through individual subsections
                        """
                        if prediction_sub[individual_sub_i] == empty_i:
                            if empty_classification_i >= updated_empty_classification_n:
                                """
                                If we have stored more than our allowed amount of empty classifications, then move on to the next one and don't store this one.
                                """
                                continue
                            """
                            Otherwise, increment if empty classification
                            """
                            empty_classification_i+=1

                        """
                        Otherwise, we store this individual subsection as the next element in our final data array.
                        """
                        #print greyscale_sub[individual_sub_i]
                        #disp_img_fullscreen(greyscale_sub[individual_sub_i])
                        import cv2
                        cv2.imshow("asdf",greyscale_sub[individual_sub_i])
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        x[total_classification_i] = greyscale_sub[individual_sub_i]
                        y[total_classification_i] = prediction_sub[individual_sub_i]

                        """
                        Then increment our global index counter
                        """
                        total_classification_i += 1
    """
    Now that we have filled our x and y arrays, open and create a new live_archive.h5 dataset.
    """
    with h5py.File(live_archive_dir, "w") as live_hf:
        live_hf.create_dataset("x", data=x)
        live_hf.create_dataset("y", data=y)
    

main(sub_h=80, 
     sub_w=145, 
     img_archive_dir="../lira/lira1/data/smol_greyscales.h5",
     predictions_archive_dir="../lira/lira1/data/predictions.h5",
     classification_metadata_dir="../slide_testing/classification_metadata.pkl",
     interactive_session_metadata_dir="interactive_session_metadata.pkl",
     live_archive_dir="../lira/lira1/data/samples/live_samples.h5")#Main execution
