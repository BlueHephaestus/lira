#For running PredictionGrids.generate() with different batch sizes to find the fastest one.
import time

import cv2
import numpy as np

from Dataset import Dataset
from base import *

#Classify then get images for the resulting predictions
dataset = Dataset()
dataset.detect_type_ones()
for mb in [23,24]:
    start = time.time()
    dataset.prediction_grids.mb_n = mb
    dataset.predict_grids()
    print("\nMB: {} s/grid: {}".format(mb, (time.time() - start)/3))
