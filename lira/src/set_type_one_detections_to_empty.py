#Take all type one detections for the given user, and set them as no detections
from Dataset import Dataset

#Load dataset
dataset = Dataset(uid="basay3", restart=False)

#Set all detections before and after editing to [] (empty)
for i in range(len(dataset.type_one_detections.before_editing)):
    dataset.type_one_detections.before_editing[i] = []
    dataset.type_one_detections.after_editing[i] = []

