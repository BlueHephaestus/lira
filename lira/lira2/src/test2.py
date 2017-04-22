import numpy as np
import h5py
a = np.memmap("test.dat", dtype="float32", mode="w+", shape=(8, 16))
with h5py.File("test.h5", mode="w", chunks=True, compression="gzip") as hf:
    #a[:] = hf.get("x")[:]
    #a[:] = hf.get("x")[:]
    hf.create_dataset("a", data=a)
    hf.create_dataset("a_shape", data=a.shape)

with h5py.File("test.h5", mode="r", chunks=True, compression="gzip") as hf:
    b_shape = tuple(hf.get("a_shape"))
    print b_shape
    b = np.memmap("test.dat", dtype="float32", mode="w+", shape=b_shape)
    b[:] = hf.get("a")[:]

print b, b.shape, type(b)
