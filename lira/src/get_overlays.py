#python3 get_overlays.py dark 8
import cv2
import numpy as np
import sys

from Dataset import Dataset
from base import *
from tqdm import tqdm

#Classify then get images for the resulting predictions
dataset = Dataset(uid=sys.argv[1], restart=False)

#Loop through predictions and images, creating a resized image for each of them.
resize_factor = 1/int(sys.argv[2])
color_key = [(255, 0, 255), (0, 0, 255), (0, 255, 0), (200, 200, 200), (0, 255, 255), (255, 0, 0), (244,66,143)]
alpha = 0.33
#k = {0:6, 1:5, 2:4, 3:0, 4:2, 5:1, 6:3}#TEMPORARY
for i, (img, prediction_grid) in enumerate(tqdm(zip(dataset.imgs, dataset.prediction_grids.after_editing), total=len(dataset.imgs))):
    #prediction_grid = dataset.prediction_grids.after_editing[k[i]]#TEMPORARY
    #Since our image and predictions would be slightly misalgned from each other due to rounding,
    #We recompute the sub_h and sub_w and img resize factors to make them aligned.
    sub_h = int(resize_factor*dataset.prediction_grids.sub_h)
    sub_w = int(resize_factor*dataset.prediction_grids.sub_w)
    fy = (prediction_grid.shape[0]*sub_h)/img.shape[0]
    fx = (prediction_grid.shape[1]*sub_w)/img.shape[1]

    #Then resize the image with these new factors
    img = cv2.resize(img, (0,0), fx=fx, fy=fy)
   
    #Make overlay to store prediction rectangles on before overlaying on top of image
    prediction_overlay = np.zeros_like(img)

    for row_i, row in enumerate(prediction_grid):
        for col_i, col in enumerate(row):
            color = color_key[col]
            #draw rectangles of the resized sub_hxsub_w size on it
            cv2.rectangle(prediction_overlay, (col_i*sub_w, row_i*sub_h), (col_i*sub_w+sub_w, row_i*sub_h+sub_h), color, -1)

    #Add overlay to image to get resulting image
    display_img = weighted_overlay(img, prediction_overlay, alpha)

    #Write img
    cv2.imwrite("../data/unit_tests/{}_{}x_overlay_{}.png".format(sys.argv[1], sys.argv[2], i), display_img)



    



