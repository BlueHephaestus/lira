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

def main(alpha=1/3.,
         sub_h=80, 
         sub_w=145, 
         img_archive_dir="../lira/lira1/data/greyscales.h5",
         predictions_archive_dir="../lira/lira1/data/predictions.h5",
         classification_metadata_dir="../slide_testing/classification_metadata.pkl"):
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
            Open a new interactive session before getting our images
            """
            interactive_session = InteractiveGUI(classifications, colors, sub_h, sub_w, alpha)
            for img_i in range(img_n):
                """
                Get our image, and associated predictions
                """
                img = img_hf.get(str(img_i))
                img_predictions = predictions_hf.get(str(img_i))

                """
                Get our number of subsections for this image from our divide factor ^ 2,
                    then start looping through.
                """
                factor = get_relative_factor(img.shape[0], None)
                sub_n = factor**2
                for sub_i in range(sub_n):
                    """
                    Now that we have an img_i and sub_i for each iteration, 
                        We check if this subsection is entirely empty, according to if all the predictions are empty. 
                        This is based on the computation-saving assumption that if all our predictions for a subsection are empty, then we have an empty subsection.
                    """
                    if not all_predictions_empty(sub_i, factor, img_predictions, classifications):
                        """
                        If it is not empty, we continue!
                        Get our img for this subsection to display from get_next_overlay_subsection()
                        As well as our predictions for this subsection using get_prediction_subsection()
                        """
                        overlay_sub = get_next_overlay_subsection(img_i, sub_i, factor, img, img_predictions, classifications, colors, alpha=1/3., sub_h=80, sub_w=145)
                        prediction_sub = get_prediction_subsection(sub_i, factor, img_predictions)

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
                                    WE NEED TO SOMEHOW UPDATE OUR IMG_PREDICTIONS FROM JUST OUR PREDICTION_SUB THAT IS STORED IN THE OVERLAY
                                    1. Update main img_predictions matrix with updated predictions subsection
                                    2. Get our new overlay subsection overlay_sub using updated img_predictions
                                    3. Reload the interactive session with new parameters
                                    4. Reset flag to false
                                """
                                img_predictions = update_prediction_subsection(sub_i, factor, img_predictions, interactive_session.predictions)
                                overlay_sub = get_next_overlay_subsection(img_i, sub_i, factor, img, img_predictions, classifications, colors, alpha=1/3., sub_h=80, sub_w=145)

                                prediction_sub = interactive_session.predictions
                                interactive_session = InteractiveGUI(classifications, colors, sub_h, sub_w, alpha)
                                interactive_session.np_img = overlay_sub
                                interactive_session.predictions = prediction_sub
                                interactive_session.start_interactive_session()

                                interactive_session.flag_refresh=False

                            if interactive_session.flag_next:
                                """
                                if our flag is set to go to the next image in our interactive session, 
                                    reset the flag, 
                                    exit our main loop, and go to the next image.
                                """
                                interactive_session.flag_next=False
                                break

                            if interactive_session.flag_quit:
                                """
                                if our flag is set to exit our session, then we also exit our main loop.
                                """
                                break

                    if interactive_session.flag_quit:
                        """
                        If we are going to the next image, we will let our loop do so. 
                        However, if we are going to quit the session, we need to exit our loops, so we break again.
                        """
                        break

                if interactive_session.flag_quit:
                    """
                    Same as above. We need to exit all of our loops if we are done with our session.
                    """
                    break


                            

                """
                start session image as nothing and set it as we go through
                """

    
    """
    We have either gone through all of our images or quit our interactive session, continue
    """



main(alpha=1/3., 
     sub_h=80, 
     sub_w=145, 
     img_archive_dir="../lira/lira1/data/smol_greyscales.h5",
     predictions_archive_dir="../lira/lira1/data/predictions.h5",
     classification_metadata_dir="../slide_testing/classification_metadata.pkl")#Main execution.
