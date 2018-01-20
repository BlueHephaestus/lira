import sys
import numpy as np
import cv2
import pickle
"""
Load suppressed rects, 
load predictions,
draw suppressed rects with thin outline or maybe overlay,
loop through predictions,
draw colored rect,
exit.
"""
classifications = ["Healthy Tissue", "Type I - Caseum", "Type II", "Empty Slide", "Type III", "Type I - Rim", "Unknown/Other"]
colors = [(255, 0, 255), (0, 0, 255), (0, 255, 0), (200, 200, 200), (0, 255, 255), (255, 0, 0), (244,66,143)]

#r = 0.031/0.2
for img_i in range(5):

    img = cv2.imread("img_%i.png" % img_i)

    with open("suppressed_rects_%i.pkl"%img_i, "r") as f:
        suppressed_rects = pickle.load(f)
        suppressed_rects = 0.2 * suppressed_rects
        suppressed_rects = suppressed_rects.astype(int)

    with open("predictions_%i.pkl"%img_i, "r") as f:
        predictions = pickle.load(f)

    sub_h = 80
    sub_w = 145
    resized_sub_h = int(np.floor(sub_h * 0.031))
    resized_sub_w = int(np.floor(sub_w * 0.031))
    fy = (resized_sub_h * predictions.shape[0]) / float(img.shape[0])
    fx = (resized_sub_w * predictions.shape[1]) / float(img.shape[1])
    resized_img = cv2.resize(img, (0,0), fx=fx, fy=fy)

    #now we have everything we need for resources - img, suppressed_rects, and predictions.
    #draw all our suppressed rects with a thinner outline than before 
    #RED outline
    alpha = 0.33
    rects = np.zeros_like(resized_img)
    overlay = np.zeros_like(resized_img)
    cv2.addWeighted(rects, alpha, resized_img, 1-alpha, 0, overlay)
    for rect in suppressed_rects:
        x1,y1,x2,y2 = rect
        resized_rect = np.array([x1*fx, y1*fy, x2*fx, y2*fy]).astype(int)
        x1,y1,x2,y2 = resized_rect
        cv2.rectangle(overlay, (x1, y1), (x2,y2), (0,0,255), 1)
        #Do this instantly, don't display until all are added

    i = 0
    #Small pause to show change
    for j in range(80):
        cv2.imwrite("images_for_gifs/%i/%07i.jpg"%(img_i,i), overlay)
        i+=1


    window_n = float(len(predictions))
    #now we loop through predictions and add transparent colored rectangle for each
    for prediction_row_i, prediction_row in enumerate(predictions):
        sys.stdout.write("\rImage {} -> {:.2%} Complete".format(img_i, i/window_n))
        sys.stdout.flush()

        #reinit overlay so we don't keep getting a progressively more faded out background, and instead retain our alpha ratio
        overlay = np.zeros_like(resized_img)
        for prediction_col_i, prediction in enumerate(prediction_row):

            color = colors[prediction]
            cv2.rectangle(rects, (prediction_col_i*resized_sub_w, prediction_row_i*resized_sub_h), (prediction_col_i*resized_sub_w+resized_sub_w, prediction_row_i*resized_sub_h+resized_sub_h), color, -1)
        #This sets overlay to have the result of adding rects to img, where the resulting overlay = alpha*rects + (1-alpha)*img, basically
        cv2.addWeighted(rects, alpha, resized_img, 1-alpha, 0, overlay)
        for rect in suppressed_rects:
            x1,y1,x2,y2 = rect
            resized_rect = np.array([x1*fx, y1*fy, x2*fx, y2*fy]).astype(int)
            x1,y1,x2,y2 = resized_rect
            cv2.rectangle(overlay, (x1, y1), (x2,y2), (0,0,255), 1)
        i+=1
        cv2.imwrite("images_for_gifs/%i/%07i.jpg"%(img_i,i), overlay)

    for j in range(80):
        cv2.imwrite("images_for_gifs/%i/%07i.jpg"%(img_i,i), overlay)
        i+=1


    
    
