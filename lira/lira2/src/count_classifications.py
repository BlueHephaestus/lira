import h5py
import numpy as np

class_n = 7

#with h5py.File("../../../lira2/data/live_samples.h5", "r") as hf:
with h5py.File("../data/augmented_samples.h5", "r", chunks=True, compression="gzip") as hf:
    x_shape = tuple(hf.get("x_shape"))
    y_shape = tuple(hf.get("y_shape"))
    x = np.memmap("x.dat", dtype="float32", mode="w+", shape=x_shape)
    y = np.memmap("y.dat", dtype="float32", mode="w+", shape=y_shape)
    x[:] = hf.get("x")[:]
    y[:] = hf.get("y")[:]

for i in range(class_n):
    print i, np.sum(y==i)

