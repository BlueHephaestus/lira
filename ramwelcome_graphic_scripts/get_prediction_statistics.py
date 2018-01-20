import pickle
import sys
import numpy as np
import cv2

classifications = ["Healthy Tissue", "Type I - Caseum", "Type II", "Empty Slide", "Type III", "Type I - Rim", "Unknown/Other"]

def p(predictions,i,n):
    print "{}, {:.2%}".format(classifications[i], np.sum(predictions==i)/float(n))
for img_i in range(5):

    with open("predictions_%i.pkl"%img_i, "r") as f:
        predictions = pickle.load(f)

    #Get proportions of each classification that is not empty slide
    n = np.sum(predictions!=3)
    print""
    print img_i
    p(predictions,1,n)
    p(predictions,5,n)
    p(predictions,2,n)
    p(predictions,4,n)
    p(predictions,0,n)
    p(predictions,6,n)



