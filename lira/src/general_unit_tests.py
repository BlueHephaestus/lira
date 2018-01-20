import os

import numpy as np
from base import *

from Dataset import Dataset
from Images import Images
from UserProgress import UserProgress
from EditingDataset import EditingDataset

#IMAGES TESTS - assumes given test images

imgs = Images()

i = 0
for img in imgs:
    i+=1
print(i==len(imgs))
print(imgs[3].shape == (1000,1000,3))
print(not np.all(imgs[0] == imgs[16]))
imgs[-1] = imgs[0]
print(np.all(imgs[0] == imgs[-1]))

#USERPROGRESS TESTS
os.remove("../data/user_progress/dark.json")
up = UserProgress("dark")

print(up.archive_fpath == "../data/user_progress/dark.json")

print(not file_exists(up.archive_fpath))
up.ensure_progress_json()
print(file_exists(up.archive_fpath))

print(not up["type_ones_finished_editing"])
print(up["type_ones_image"]==0)
print(not up.editing_started())

up["type_ones_finished_editing"] = True
up["type_ones_image"]=42

print(up["type_ones_finished_editing"])
print(up["type_ones_image"]==42)

print(up.editing_started())

up.restart()
print(not up["type_ones_finished_editing"])
print(up["type_ones_image"]==0)

#TYPEONEDETECTIONS TESTS

#PREDICTIONGRIDS TESTS

#EDITINGDATASET TESTS
d = Dataset()
ds = EditingDataset(d, "dark", "/tmp/")
i = 0
for img in imgs:
    ds[i] = img
    i+=1
print(i==len(ds))
print(not np.all(ds[0]==ds[16]))
ds[0] = ds[16]
print(np.all(ds[0]==ds[16]))








