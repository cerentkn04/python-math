import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

object_img = cv.imread('photo3.jpeg')
scene_img= cv.imread('photo4.jpeg')


sift = cv.SIFT_create()


kp1, des1 = sift.detectAndCompute(object_img, None)
kp2, des2 = sift.detectAndCompute(scene_img, None)


FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=1000)


flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

num_keypoints = len(good_matches)
result = {
    "status": "PASS" if num_keypoints >= 1000 else "FAIL",
    "keypoints_detected": num_keypoints
}

img_matches = cv.drawMatches(object_img, kp1, scene_img, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imshow("show",img_matches)
print(result)




cv.waitKey(0)
cv.destroyAllWindows()