import numpy as np
from PIL import Image
import cv2 as cv

# Open and resize image using PIL
image = Image.open("sample2.jpg")
image = image.resize((500, 500))
ddepth = cv.CV_16S

# Convert PIL image to NumPy array (OpenCV format)
image = np.array(image)

# Convert RGB to BGR for OpenCV
image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

# Show the original image (now in OpenCV format)
cv.imshow("Original Image", image)

# Define a 5x5 blur kernel
kernel = np.ones((5, 5), np.float32) / 25
image_gray=cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Apply filter2D for blurring
blur = cv.filter2D(image, -1, kernel)
Gblur = cv.GaussianBlur(image,(5,5),21)
LaplacePic = cv.Laplacian(image_gray, ddepth, 3)
abs_dst = cv.convertScaleAbs(LaplacePic)
# Show the blurred image
cv.imshow("Blurred Image", blur)
cv.imshow("gaussian Image", Gblur)
cv.imshow("Laplace Image", abs_dst)

# Wait for a key press to close windows
cv.waitKey(0)
cv.destroyAllWindows()
