import numpy as np
import cv2

theta = 90
sx = .5
sy = .5
dx = 32
dy = 128
w = 512
h = 512

theta = theta * np.pi / 180.#Degrees -> Radians
rotation = np.float32([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0,0,1]])
scaling = np.float32([[sx,0,0],[0,sy,0],[0,0,1]])
translation = np.float32([[1,0,dx],[0,1,dy],[0,0,1]])

"""
Moves top left corner of frame to center of frame
"""
top_left_to_center = np.array([[1., 0., .5*w], [0., 1., .5*h], [0.,0.,1.]])

"""
Moves center of frame to top left corner of frame, the inverse of our previous transformation
"""
center_to_top_left = np.array([[1., 0., -.5*w], [0., 1., -.5*h], [0.,0.,1.]])

T = scaling.dot(translation).dot(rotation)
T = top_left_to_center.dot(T).dot(center_to_top_left)

print T
a = np.floor(np.random.rand(512, 512, 3)*255).astype(np.uint8)
a[350:380, 350:380] = [0, 0, 0]

b = cv2.warpAffine(a, T[:2], (512, 512), borderValue=[244, 244, 244])
cv2.imshow("asdf", a)
cv2.waitKey(0)
cv2.imshow("asdf", b)
cv2.waitKey(0)

