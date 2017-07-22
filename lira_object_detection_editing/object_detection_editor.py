"""
"""
import pickle
import cv2
import numpy as np

def disp_img_fullscreen(img, name="test"):
    """
    Displays the given image full screen. 
    Usually used for debugging, uses opencv's display methods.
    """
    cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)          
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, 1)
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
loop through and load test cases
"""
resize_factor = 0.1
for i in range(8):
    print i
    with open("%i.pkl" % i, "r") as f:
        rects = pickle.load(f)
    rects = np.array(rects) #since it's a static size now and we need elementwise scalar multiplications
    rects = rects * resize_factor#resize to match our image
    rects = rects.astype(int)
    img = cv2.imread("testcase_%i.png" % i)
    #img = cv2.resize(img, (0,0), fx=resize_factor, fy=resize_factor)
    """
    for (x1,y1,x2,y2) in rects:
        cv2.rectangle(img, (x1,y1),(x2,y2), (0,0,255), 2)
    #At this point image is ready for disp
    """
    #disp_img_fullscreen(img)
    #now display rects on img with rectangle method
"""
"""
