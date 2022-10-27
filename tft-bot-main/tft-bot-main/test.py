import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('trial2.png',0)
img2 = img.copy()
template = cv.imread('loadingscreen.png',0)
w, h = template.shape[::-1]
# All the 6 methods for comparison in a list
method = cv.TM_CCOEFF_NORMED

img = img2.copy()
# Apply template Matching
res = cv.matchTemplate(img,template,method)
threshold = 0.8
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

cv.imwrite('res.png',img)