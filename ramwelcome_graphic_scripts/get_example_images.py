import numpy as np
import h5py
import cv2
img_idxs = [0, 13, 17, 22, 25]
with h5py.File("../lira/lira1/data/images.h5", "r", chunks=True, compression="gzip") as hf:
    for i, img_i in enumerate(img_idxs):
        img = np.array(hf.get(str(img_i)))
        r = 0.031
        img = cv2.resize(img, (0,0), fx=r, fy=r)
        cv2.imwrite("img_%i.png" % i, img)
