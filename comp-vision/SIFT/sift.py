import cv2
import numpy as np
import matplotlib.pyplot as plt
from Sift_Operations import *

image1 = cv2.imread('eif1.jpg') 
image2 = cv2.imread('eif2.jpg') 


height, width = image1.shape[:2]
image2 = cv2.resize(image2, (width, height)) 


Image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
Image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)


sift = cv2.xfeatures2d.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(image1,None)
keypoints_2, descriptors_2 = sift.detectAndCompute(image2,None)
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(descriptors_1,descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)
img3 = cv2.drawMatches(image1, keypoints_1, image2, keypoints_2, matches[:50], image2, flags=2)
plt.imshow(img3),plt.show()



