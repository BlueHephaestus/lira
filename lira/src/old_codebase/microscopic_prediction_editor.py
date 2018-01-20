"""
This file is our main file for our microscopic prediction editing,
    where all of our child classes/functions get referenced and used for the entire process 
    of loading, displaying, updating, saving, and retraining.

Further documentation can be found below throughout the file and in the main function, edit_microscopic_predictions().
-Blake Edwards / Dark Element
"""

import sys, h5py, pickle
import numpy as np

import PIL
from PIL import Image

sys.path.append("../lira_static/")

import img_handler
from img_handler import *

import subsection_handler
from subsection_handler import *

import microscopic_interactive_gui_handler
from microscopic_interactive_gui_handler import *

def edit_microscopic_predictions(sub_h=80, 
         sub_w=145, 
         img_archive_dir="../lira/lira1/data/images.h5",
         resized_img_archive_dir="../lira/lira1/data/resized_images.h5",
         predictions_archive_dir="../lira/lira1/data/predictions.h5",
         classification_metadata_dir="../lira_static/classification_metadata.pkl",
         interactive_session_metadata_dir="interactive_session_metadata.pkl",
         live_archive_dir="../lira/lira2/data/live_samples.h5",
         dual_monitor=False,
         resize_factor=0.1,
         save_new_samples=False,
         images_already_resized=False,
         rgb=True):

    """
    Arguments:
        sub_h, sub_w: The size of our individual subsections in our images. This will be resized with resize_factor.
        img_archive_dir: Filepath of our images. Will have predictions overlaid for Interactive GUI.
        resized_img_archive_dir: a string filepath of the .h5 file where the resized images will be stored.
        predictions_archive_dir: Filepath of our predictions. Will be overlaid on images for Interactive GUI.
        classification_metadata_dir: Filepath of our classification metadata. Will be used for handling our predictions / classifications properly.
        interactive_session_metadata_dir: Filepath of our interactive session metadata. 
            Will be used for loading/storing user-modifiable parameters in Interactive GUI.
        live_archive_dir: Filepath of our live samples archive. Will be used to store the samples obtained through our interactive session. Updated completely at the end of each session.
        dual_monitor: Boolean for if we are using two monitors or not. 
            Shrinks the width of our display if we are, and leaves normal if not.
        resize_factor: The scale to resize our image by, e.g. 0.5 will make a 400x400 image into a 200x200 image.
            This should always be 0 < resize_factor <= 1.0
        save_new_samples: Boolean for if we should save all our new labeled data in an archive.
            While it will automatically keep track of the new predictions as part of the process,
                and update the original predictions archive, 
                it won't save their associated image subsections / inputs.
            While it would be possible to put this in a separate file, by putting it in this file:
                we would not be able to save the samples if we quit mid-session, 
                we would not be able to save them automatically,
                and we would not be able to balance the empty slide samples to the number of other samples easily.
        images_already_resized: Boolean for if we've already resized our images before. Defaults to False.
        rgb: Boolean for if we are handling rgb images (True), or grayscale images (False).

    Returns:
        This is our main execution for LIRA-Live. Here, we handle all our sub-programs and scripts for LIRA-Live.
        After opening our archives,
        1. We check to see if we need to resize all our images again, i.e. if we stopped mid-session last time. If we do need to resize:
            a. We loop through each image,
            b. Resize them using our resize_factor, sub_h, and sub_w to get it resized just the right amount,
            c. And save the images to resized_img_archive_dir
        2. Afterwards, we loop through each image index, with which we can get the entire image and the assorted predictions.
        3. We then generate an overlay from our predictions to overlay on top of our image, and open our interactive GUI session.
        4. The user uses our interactive GUI to modify our predictions, checking to correct any mistakes they notice. 
            As they correct these, the predictions are saved after they go to the next image.
        5. They can quit mid-session to open the session again later, and we will save their progress along with the metadata such as alpha transparency for the session.
        6. Once they have gone through all the images, we save the predictions for the last image.
        7. 
            a. If the option for save_new_samples=True, then we save all our predictions along with associated subsection inputs (at full resolution, of course)
                for training another model later.
            b. If not, we skip the above step and we are done.
        You can view further details in the documentation below.
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
    prev_img_i, alpha = pickle.load(f)
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
    We resize our subsections before resizing our image, since we will use them to calculate how much to resize our image.
        (or resized, if we already did this and we are re-opening the GUI to continue our session)
    Explanation for this found in the resizing section below.
    """
    resized_sub_h = int(resize_factor * sub_h)
    resized_sub_w = int(resize_factor * sub_w)

    """
    If we've already resized our images, we have no need to do this.
    So we check if we were already mid-session before running this, i.e. if our previous img_i is >= 0.
    If so, then we don't need to resize any images because we already did that last time.
    Otherwise (i.e. previous img_i is < 0), we need to resize them.

    For now, it's necessary to resize again even if already done for the earlier UI, as this resizes based on the subsection size.
    """
    if prev_img_i < 0 or not images_already_resized:
        """
        Open our image archive, 
            loop through all the images in it, and resize them down with our resize_factor argument.
        Using these resized images, we create a new resized image archive for use with our interactive GUI.
        Note: we open our predictions archive to use in calculating how much to resize our image. Explanation below.
        """
        print "Resizing Images..."
        with h5py.File(img_archive_dir, "r", chunks=True, compression="gzip") as img_hf:
            with h5py.File(resized_img_archive_dir, "w", chunks=True, compression="gzip") as resized_img_hf:
                with h5py.File(predictions_archive_dir, "r") as predictions_hf:
                    """
                    Get image number for iteration
                    """
                    img_n = len(img_hf.keys())

                    """
                    We want to save time by not re-initializing a big block of memory for our image
                        every time we get a new image, so we find the maximum shape an image will be here.
                    We do this by the product since blocks of memory are not in image format, their size 
                        only matters in terms of raw number.
                    We do the max shape because we initialize the img block of memory, and then initialize a bunch of other
                        blocks of memory around this in the loop. 
                    This means if we had less than the max, and then encountered a larger img than our current allocated block of memory,
                        it would have to reallocate. 

                    As my friend explains: "[Let's say] you initialize to the first image. Then you allocate more memory for other things inside the loop. That likely "surrounds" the first image with other stuff. Then you need a bigger image. Python can't realloc the image in place, so it has to allocate a new version, copy all the data, then release the old one."
                    So we instead initialize it to the maximum size an image will be.
                    """
                    max_shape = img_hf.get("0").shape
                    for img_i in range(img_n):
                        img_shape = img_hf.get(str(img_i)).shape
                        if np.prod(img_shape) > np.prod(max_shape):
                            max_shape = img_shape

                    """
                    Now initialize our img block of memory to to this max_shape, 
                        with the same data type as our images
                    """
                    img = np.zeros(max_shape, img_hf.get("0").dtype)

                    """
                    Start looping through images
                    """
                    for img_i in range(img_n):
                        #Print progress
                        print "Resizing Image %i/%i..." % (img_i, img_n-1)

                        """
                        Get our image by resizing our block of memory instead of deleting and recreating it,
                            and make sure to read directly from the dataset into this block of memory,
                            so that we can reference it quickly when iterating through subsections for classification later.
                        """
                        img_dataset = img_hf.get(str(img_i))
                        img.resize(img_dataset.shape)
                        img_dataset.read_direct(img)

                        """
                        Get our predictions
                        """
                        predictions = predictions_hf.get(str(img_i))

                        """
                        The reason we do the below is because of the following problem. 
                       
                        Each time we draw a resized subsection rectangle, it will be of size
                            resized_sub_h x resized_sub_w. 
                        
                        Since we draw the same amount of rectangles as predictions,
                            we will have predictions.shape[0] x predictions.shape[1] rectangles,
                            each one of shape resized_sub_h x resized_sub_w.

                        So we know that:
                            the height of our overlay is resized_sub_h * predictions.shape[0]
                            the width of our overlay is  resized_sub_w * predictions.shape[1]

                        However since we have to cast our resized_sub_h to integer in the calculation,
                            it means we may slowly end up with an overlay with dimensions which
                            don't match our image. This causes a mismatch between the images 
                            once the entire overlay is created, and it's very ugly. 

                        Fortunately we can fix this. Using the height as the example dimension:

                            img.shape[0] * fy = resized_sub_h * predictions.shape[0]

                        We can see this is the condition we want to be true. 
                        We want the resized dimension(s) of our original image to match
                            the dimension(s) of our prediction overlay.
                        
                        So from this, we can see we already know everything but fy, 
                            and since it's much easier to solve for fy than to figure out some
                            special way to draw our subsection rectangles, we calculate:

                            fy = (resized_sub_h * predictions.shape[0])/img.shape[0]

                        And we repeat that for fx, as well.
                        """
                        fy = (resized_sub_h * predictions.shape[0]) / float(img.shape[0])
                        fx = (resized_sub_w * predictions.shape[1]) / float(img.shape[1])
                        resized_img = cv2.resize(img, (0,0), fx=fx, fy=fy)

                        """
                        And write it to our resized img dataset.
                        """
                        resized_img_hf.create_dataset(str(img_i), data=resized_img)

    """
    Now that our resized image archive is ready, we open it for our images to use for displaying in our UI,
        as well as our predictions archive for both reading our current predictions and writing new ones.
    """
    with h5py.File(predictions_archive_dir, "r+") as predictions_hf:
        with h5py.File(resized_img_archive_dir, "r") as img_hf:
            """
            Get the number of images to loop through
            """
            img_n = len(img_hf.keys())
            
            """
            Open a new interactive session before starting our loops.
                and then start looping from where we left off last to our img_n
            """
            interactive_session = InteractiveGUI(classifications, colors, resized_sub_h, resized_sub_w, alpha, dual_monitor)

            """
            We want to save time by not re-initializing a big block of memory for our image
                every time we get a new image, so we find the maximum shape an image will be here.
            We do this by the product since blocks of memory are not in image format, their size 
                only matters in terms of raw number.
            We do the max shape because we initialize the img block of memory, and then initialize a bunch of other
                blocks of memory around this in the loop. 
            This means if we had less than the max, and then encountered a larger img than our current allocated block of memory,
                it would have to reallocate. 

            As my friend explains: "[Let's say] you initialize to the first image. Then you allocate more memory for other things inside the loop. That likely "surrounds" the first image with other stuff. Then you need a bigger image. Python can't realloc the image in place, so it has to allocate a new version, copy all the data, then release the old one."
            So we instead initialize it to the maximum size an image will be.
            """
            max_shape = img_hf.get("0").shape
            for img_i in range(img_n):
                img_shape = img_hf.get(str(img_i)).shape
                if np.prod(img_shape) > np.prod(max_shape):
                    max_shape = img_shape

            """
            Now initialize our img block of memory to to this max_shape, 
                with the same data type as our images
            """
            img = np.zeros(max_shape, img_hf.get("0").dtype)

            """
            Start looping through images
            """
            for img_i in range(img_n):
                #Print progress
                print "Image %i/%i..." % (img_i, img_n-1)

                """
                We then read in our predictions.

                An important note on predictions in LIRA-Live:
                We have 2 problems:
                    1. We want to display only one color in our gui for each prediction, even though each entry is a vector of probabilities
                    2. We also need to retain the link to our predictions_hf.get, so that
                        predictions in the file can be easily updated once they are corrected using the GUI tool.
                So if we just normally argmaxed over img_predictions, we'd break #2 and no longer have a link back to our file.
                But if we left it as is, we'd not have an easy way to display the predictions.

                The solution I came up with was to create a copy of our predictions dataset, 
                    and argmax over this copy to display / generate our overlay. 
                    Once we have new predictions via our UI, we convert the predictions to one-hots (essentially the approximate inverse of an argmax) 
                    and then update our original predictions using this.
                """
                #Get dataset
                predictions_dataset = predictions_hf.get(str(img_i))

                #Create copy
                predictions = np.zeros_like(predictions_dataset)
                predictions[...] = predictions_dataset[...]

                #Argmax predictions
                predictions = np.argmax(predictions, axis=2)

                """
                We don't do anything if we are not yet to our last stopping point / where we left off.
                If we aren't there yet, we skip all the unnecessary stuff, only adding our classification data.
                Then, we go to the next one.
                We can't just start immediately on the image we left off on because we need to keep track of our 
                    empty_classification_n and total_classification_n numbers.
                """
                if img_i < prev_img_i:
                    empty_i = list_find(classifications, "Empty Slide")
                    empty_classification_n += np.sum(predictions==empty_i)
                    total_classification_n += predictions.size
                    continue

                """
                If we are on the right image, we can continue.

                Get our full image by resizing our block of memory instead of deleting and recreating it,
                    and make sure to read directly from the dataset into this block of memory,
                    so that we can reference it quickly when iterating through subsections for classification later.
                """
                img_dataset = img_hf.get(str(img_i))
                img.resize(img_dataset.shape, refcheck=False)
                img_dataset.read_direct(img)

                """
                Now that we have the correct image, we initialize our new color overlay to match our img
                """
                overlay = np.zeros_like(img)

                """
                Then loop through each prediction in our predictions,
                    using the row and col indices with the resized_sub_h and resized_sub_w for the location of our colored rectangle on our overlay,
                    and use the prediction's value in our color key for the color of this rectangle.
                """
                fill_overlay(overlay, predictions, resized_sub_h, resized_sub_w, colors)

                """
                Then overlay the overlay onto our image, using our alpha parameter
                    We also re-initialize our combined overlay to zeros every time we do this,
                        so as to avoid overlaying on top of ourself.
                """
                combined_overlay = add_weighted_overlay(img, overlay, alpha, rgb=True)

                """
                Set our interactive UI's predictions and image before starting the session and our main ui loop
                """
                interactive_session.np_img = combined_overlay
                interactive_session.predictions = predictions
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
                        predictions = interactive_session.predictions

                        """
                        2. Get our new overlay using our updated predictions
                        """
                        fill_overlay(overlay, predictions, resized_sub_h, resized_sub_w, colors)
                        
                        """
                        3. Update our overlay using this and our updated alpha parameter
                            We also re-initialize our combined overlay to zeros every time we do this,
                                so as to avoid overlaying on top of ourself.
                        """
                        combined_overlay = add_weighted_overlay(img, overlay, alpha, rgb=True)

                        """
                        4. Reload the interactive session with new parameters
                        """
                        interactive_session = InteractiveGUI(classifications, colors, resized_sub_h, resized_sub_w, alpha, dual_monitor)
                        interactive_session.np_img = combined_overlay
                        interactive_session.predictions = predictions

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
                        predictions = interactive_session.predictions
                        
                        """
                        2. Reset flag to false
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
                        predictions = interactive_session.predictions
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
                Now that we are done with classifying this image (or it was skipped because it was entirely empty),
                    we increment our total_classification_n and empty_classification_n accordingly

                We know that if we quit on this subsection, this code will not be executed. 
                It will only be executed if the user has selected to go to the next one, meaning they are done with this image.

                1. Get number of empty classifications via same method to check if they were all empty.
                """
                empty_i = list_find(classifications, "Empty Slide")
                empty_classification_n += np.sum(predictions==empty_i)

                """
                2. Get number of total classifications via the raw number of items in our predictions, now that we are finished with it.
                """
                total_classification_n += predictions.size

                """
                3. Update main predictions_dataset archive with updated predictions subsection.
                    As mentioned earlier, our dataset is expected to be of shape h, w, class_n,
                    so we need to first convert each prediction integer to a one-hot vector of length class_n.
                    
                    In order to do this, we will flatten it, using our to_categorical function, then reshape it back into a matrix.
                    Afterwards we will set all our datasets elements to match those of our predictions, 
                        so that we can write the changes in predictions to our actual archive.

                """
                #Get categorical
                class_n = predictions_dataset.shape[-1]
                predictions = to_categorical(predictions.flatten(), class_n)
                #Reshape to same shape as our original dataset
                predictions = np.reshape(predictions, predictions_dataset.shape)
                #Set dataset / archive to this
                predictions_dataset[...] = predictions[...]

        """
        At this point, we have either gone through all of our images or quit our interactive session.
            If we went through all our images, we already have updated our predictions in the archive
            And already have our updated metadata. 
        So we don't need to worry about getting our updated stuff from the interactive session.

        We also close our image archive and leave our predictions one open.
        """
        """
        1. Delete our interactive session to save memory, we will need as much memory as we can get for the next part.
        """
        del interactive_session

        """
        Then we can continue to updating metadata:

        Now, (since we already updated our copies on disk of predictions) we update any metadata for this session.

        We do this by writing the img_i and alpha, regardless of where they left off, so that the user can continue where they left off if opening a session again.
        
        Note: We do nothing special if img_i = img_n-1 so that we don't accidentally restart the session,
            and only restart the session if we intentionally choose to via resetting the metadata or other methods.
        """
        print "Session ended. Updating and storing updated metadata..."
        with open(interactive_session_metadata_dir, "w") as f:
            #Quit the session
            pickle.dump((img_i, alpha), f)

        if save_new_samples:
            """
            And now, all that is left is to save all of our updated samples for use in the next training session (if we chose to).
            In order to do this, we have to create two new np arrays to store them in, and also store them in the following format:
                X dataset shape = (sample_n, sub_h*sub_w) -> the flattened images
                Y dataset shape = (sample_n, ) -> the integer classification indices for each image.
            This is rather complex. So let's take it step by step.

            Modify our indices so that we only reference the images we have updated (the images behind the last modified one), 
                unless we have run out of images (img_i == img_n-1 && sub_i == sub_n-1), 
                in which case we get everything, including our most recently modified image.

            We also open the original full resolution images, since we're going to need them to get the full resolution.
            """
            with h5py.File(img_archive_dir, "r", chunks=True, compression="gzip") as img_hf:
                print "Updating Live Samples Archive for Training..."

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
                    with dtype np.uint8 so we can display them properly for testing
                We also handle our rgb option here.
                """
                if rgb:
                    x = np.zeros((updated_total_classification_n, sub_h*sub_w, 3), dtype=np.uint8)
                else:
                    x = np.zeros((updated_total_classification_n, sub_h*sub_w), dtype=np.uint8)
                y = np.zeros((updated_total_classification_n,), dtype=np.uint8)

                """
                Loop through large subsection, img, and then individual subsection. We go all the way to individual to ensure we don't add any more empty classifications than needed.
                    Since our img_i is only relevant for the last large subsection we run through, we have to loop in a rather peculiar fashion.
                    We also have to make sure to loop in the same manner as our original loop for the interactive session.
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

                """
                We then load each image the same way we did earlier, unfortunately we can't use our data from earlier to skip a lot of the work
                    because we might not have needed to resize and thus don't actually have it. 
                Also we aren't looping through all the images, just the ones up until the image before our last opened image, which is image img_i.
                However, in the case that we got to the final image in our set, there will not be another image after that, so we assume they finished classifying this last image,
                    So we do img_n = img_i + 1 in that case, and img_n = img_i in all other cases.
                """
                if img_i == img_n-1:
                    img_n = img_i + 1
                else:
                    img_n = img_i

                max_shape = img_hf.get("0").shape
                for img_i in range(img_n):
                    img_shape = img_hf.get(str(img_i)).shape
                    if np.prod(img_shape) > np.prod(max_shape):
                        max_shape = img_shape

                """
                Now initialize our img block of memory to to this max_shape, 
                    with the same data type as our images
                """
                img = np.zeros(max_shape, img_hf.get("0").dtype)

                """
                Start looping through images
                """
                for img_i in range(img_n):
                    #Print progress
                    print "Image %i/%i..." % (img_i, img_n-1)

                    """
                    Get our image by resizing our block of memory instead of deleting and recreating it,
                        and make sure to read directly from the dataset into this block of memory,
                        so that we can reference it quickly when iterating through subsections for classification later.
                    """
                    img_dataset = img_hf.get(str(img_i))
                    img.resize(img_dataset.shape, refcheck=False)
                    img_dataset.read_direct(img)

                    """
                    This time we aren't writing anything to our predictions archive, so we argmax immediately.
                        We then flatten it since that's the exact format we want our predictions to be in for our final samples archive.
                    """
                    predictions_dataset = predictions_hf.get(str(img_i))
                    predictions = np.argmax(predictions_dataset, axis=2)
                    predictions = predictions.flatten()

                    """
                    We can easily get the number of predictions in our subsection to get the number
                        of total predictions and subsequently individual subsections we have in this subsection.
                    """
                    individual_sub_n = predictions.size

                    """
                    Then, since our image is too large to immediately convert to a matrix of it's component subsections, we loop through them using our
                        get_subsections generator.

                    We then place each subsection in our resulting sample arrays.

                    Also important note that we are using full resolution, because these are samples for models which expect full resolution sub_h x sub_w input.
                    """
                    for individual_sub_i, sub in enumerate(get_subsections(sub_h, sub_w, img, rgb=rgb)):
                        """
                        And start looping through individual subsections via our generator
                        """
                        if predictions[individual_sub_i] == empty_i:
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
                            Checking to see if it's rgb so we know how to reshape it.
                        """
                        if rgb:
                            sub = np.reshape(sub, (11600, 3))
                        else:
                            sub = sub.flatten()
                        x[total_classification_i] = sub
                        y[total_classification_i] = predictions[individual_sub_i]

                        """
                        Then increment our global index counter
                        """
                        total_classification_i += 1
                """
                Now that we have filled our x and y arrays, open and create a new live archive dataset.
                """
                with h5py.File(live_archive_dir, "w", chunks=True, compression="gzip") as live_hf:
                    live_hf.create_dataset("x", data=x)
                    live_hf.create_dataset("x_shape", data=x.shape)
                    live_hf.create_dataset("y", data=y)
                    live_hf.create_dataset("y_shape", data=y.shape)

