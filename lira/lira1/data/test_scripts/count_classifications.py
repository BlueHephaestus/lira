import h5py
import numpy as np

class_n = 7

with h5py.File("live_samples.h5", "r") as hf:
    x = np.array(hf.get("x"))
    y = np.array(hf.get("y"))

for i in range(class_n):
    print i, np.sum(y==i)

