import sys

sys.path.append("lira/lira2/src")

<<<<<<< HEAD
import train_microscopic_model
=======
import lira2_pre_transfer_learning_mk2
>>>>>>> 0b4dd0478ac0b5eecfde8789898381e427581835

sys.path.append("lira_type_1_detection")

import train_detection_model

def train_models(model_title, detection_model_title):
    """
    Arguments:
        model_title: Name of the microscopic model we're training.
        detection_model_title: Name of the macroscopicmodel we're training.
    Returns:
        Uses the model_title and detection_model_title given to get a filename for our models, and trains both of them using their respective samples.
    """
    """
    From our samples file, train our models and save in saved_networks/`nn`
        You need to change your input dimensions in the model training file, if you want to switch between grayscale/rgb
    """
    print "Training Models..."
    """
    Get filenames
    """
    model = model_title.lower().replace(" ", "_")
    model1 = model + "_model_1"
    model2 = model + "_model_2"
    detection_model = detection_model_title.lower().replace(" ", "_")

    """
    Train our macroscopic model with our samples for this model
    """
<<<<<<< HEAD
    #train_detection_model.train_model("lira_type_1_detection/augmented_samples.h5", detection_model)
=======
    #train_detection_model.train_model("augmented_samples.h5",detection_model)
>>>>>>> 0b4dd0478ac0b5eecfde8789898381e427581835

    """
    Train our first microscopic model on our samples for this model
    """
<<<<<<< HEAD
    #train_microscopic_model.train_model(model1, model_dir="lira/lira2/saved_networks", archive_dir="lira/lira2/data/augmented_model_1_samples.h5")
=======
    #lira2_pre_transfer_learning_mk2.train_model(model1, model_dir="lira/lira2/saved_networks", archive_dir="lira/lira2/data/augmented_model_1_samples.h5")
>>>>>>> 0b4dd0478ac0b5eecfde8789898381e427581835

    """
    Train our second microscopic model on our samples for this model
    """
<<<<<<< HEAD
    #train_microscopic_model.train_model(model2, model_dir="lira/lira2/saved_networks", archive_dir="lira/lira2/data/augmented_model_2_samples.h5")
=======
    lira2_pre_transfer_learning_mk2.train_model(model2, model_dir="lira/lira2/saved_networks", archive_dir="lira/lira2/data/augmented_model_2_samples.h5")
>>>>>>> 0b4dd0478ac0b5eecfde8789898381e427581835

train_models("LIRA MK2.8 Cooperative Model Classification", "Type1 Detection Model MK5")
