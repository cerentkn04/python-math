import cv2
import numpy as np
import matplotlib.pyplot as plt
def extract_sift_features(img):
    sift_initialize = cv2.SIFT_create()  # Use SIFT_create() for OpenCV 4.x
    key_points, descriptors = sift_initialize.detectAndCompute(img, None)
    return key_points, descriptors

# Function to display SIFT features
def showing_sift_features(img, key_points):
    return cv2.drawKeypoints(img, key_points, None)