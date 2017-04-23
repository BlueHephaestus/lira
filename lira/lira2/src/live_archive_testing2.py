import h5py
import numpy as np
import sys
import cv2

sys.path.append("../../../lira_static/")

import img_handler
from img_handler import *

with h5py.File("../data/live_samples.h5", "r") as hf:
    x = np.array(hf.get("x"))
    y = np.array(hf.get("y"))

print x.shape, y.shape
"""
OI FUTURE SELF
    We don't have problems in saving them in the right shape, and those calculations.
    We do however seem to have problems with how they are saved, however.
    Upon printing each subsection, we get far too many that are empty, and the ones that are not seem to be completely black, when they should be our greyscales.
    Upon printing all predictions, we do not get the same distribution of predictions expected. 
    We need to:
        1. check to see if there are a matching number of empty slide classifications, 128==128, YES
        2. record the indices: classifications key, DONE
        3. and find what predictions we actually do have to hopefully gain some insight. DONE

    Alright insight gained. So it's only assigning values 135 different times, and since we initialize predictions to zeros, we end up with the remainder being zeros, resulting in our crazy distribution of predictions.
    So we need to find out why it's only assigning values 135 different times, instead of 35+128 and 734 times (total: 897).
    It was because we were skipping empty classifications incorrectly. FIXED
    0 16 - absolutely correct
    1 43
    2 233
    3 128 - absolutely correct
    4 412
    5 68

    i'm not checking the rest, I just know that I placed 16 0s on purpose, so it is good that we should have those nailed. time to display the subsections to see if those are correct.

    Alright the subsections aren't correct woo

    upon printing them during execution, they are thin as fuck. I think i'm going to leave it at that, you should be able to fix it no problem.
    Oh also get started on hw today future self. maybe call mom if you want. I'm going to bed now because that is enough.
    Good shit, good luck, and have fun. 

    we fixed one bug, however these seem to not be printing correctly, still.
    fixed the other bug too.
"""
#a = x[:167]
x = np.reshape(x, (-1, 80, 145))
classifications = ["Healthy Tissue", "Type I - Caseum", "Type II", "Empty Slide", "Type III", "Type I - Rim", "Unknown"]

#BGR Colors to give to each classification index
#         Pink,          Red,         Green,       Light Grey,      Yellow,        Blue         Purple
colors = [(255, 0, 255), (0, 0, 255), (0, 255, 0), (200, 200, 200), (0, 255, 255), (255, 0, 0), (244,66,143)]
print x.shape
for i, a in enumerate(x):
    if y[i] != 3:
        print classifications[y[i]]
        cv2.imshow("asdf",a)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
#print y[:167]
#y[:166]
