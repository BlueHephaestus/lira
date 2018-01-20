import sys
import numpy as np
import h5py
import pickle

sys.path.append("../lira_static")

import object_detection_handler
from object_detection_handler import ObjectDetector

detection_model_title = "Type1 Detection Model MK5"
detection_model = detection_model_title.lower().replace(" ", "_")
object_detector = ObjectDetector(detection_model, "../lira/lira2/saved_networks")
img_idxs = [0, 13, 17, 22, 25]
with h5py.File("../lira/lira1/data/images.h5", "r", chunks=True, compression="gzip") as hf:
    for i, img_i in enumerate(img_idxs):
        img = np.array(hf.get(str(img_i)))
        #Set to not use suppression
        bounding_rects = object_detector.generate_bounding_rects(img)
        bounding_rects = np.array(bounding_rects)
        with open("unsuppressed_rects_%i.pkl"%i, "w") as f:
            pickle.dump(bounding_rects, f)


