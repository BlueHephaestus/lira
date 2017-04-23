import numpy as np
import h5py
with h5py.File("../data/rgb_rim_samples.h5", "r") as hf:
    x1 = np.array(hf.get("x"))
    y1 = np.array(hf.get("y"))

with h5py.File("../data/rgb_samples.h5", "r") as hf:
    x2 = np.array(hf.get("x"))
    y2 = np.array(hf.get("y"))

print x1.shape, x2.shape
print y1.shape, y2.shape
x = np.concatenate((x1, x2), axis=0)
y = np.concatenate((y1, y2), axis=0)
print x.shape, y.shape

with h5py.File("../data/test_samples.h5", "w", chunks=True, compression="gzip") as hf:
    hf.create_dataset("x", data=x)
    hf.create_dataset("x_shape", data=x.shape)
    hf.create_dataset("y", data=y)
    hf.create_dataset("y_shape", data=y.shape)

