import numpy as np
import h5py
import pickle
img_idxs = [0, 13, 17, 22, 25]
with h5py.File("../lira/lira1/data/test_predictions.h5", "r") as hf:
    for i, img_i in enumerate(img_idxs):
        predictions = np.array(hf.get(str(img_i)))
        print predictions.shape
        predictions = np.argmax(predictions, axis=2)
        predictions = predictions.astype(np.uint8)
        with open("predictions_%i.pkl"%i, "w") as f:
            pickle.dump(predictions, f)
#ok so we need to get this archive using our last? one???? no we should actually fix the pipeline and test because we might have better results tbh
