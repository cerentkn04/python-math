import numpy as np
from PIL import Image
import cv2 as cv


image = Image.open("sample.jpg")
image = image.resize((500, 500))
ddepth = cv.CV_16S

image = np.array(image)

image = cv.cvtColor(image, cv.COLOR_RGB2BGR)


cv.imshow("Original Image", image)



kernel = np.ones((10, 10), np.float32) / 100
kernel1 = np.ones((5, 5), np.float32)/30
kernel1y = kernel1.T

image_gray=cv.cvtColor(image, cv.COLOR_BGR2GRAY)
normalizedimg =cv.normalize(image, None, 0, 1, cv.NORM_MINMAX, dtype=cv.CV_32FC1)
filtered_x = cv.filter2D(normalizedimg,ddepth= -1, kernel =kernel1)
filtered_y = cv.filter2D(normalizedimg, ddepth= -1, kernel =kernel1y)

theta_deg = 45
m=200
theta = np.radians(theta_deg)

x_index, y_index = np.meshgrid(np.arange(-m, m+1), np.arange(-m, m+1))



G = lambda x, y: np.exp(-(x**2 + y**2))
G0 = lambda x, y: -2*x*G(x,y)
G90 = lambda x, y: 2*y*G(x,y)

gk = G(x_index, y_index)
gk0 = G0(x_index, y_index)
gk90 = G90(x_index, y_index)


gk_theta = np.cos(theta)*gk0 + np.sin(theta)*gk90

blur = cv.filter2D(image, -1, kernel)
LaplacePic = cv.Laplacian(image_gray, ddepth, 3)
abs_dst = cv.convertScaleAbs(LaplacePic)

R_Theta = cv.filter2D(normalizedimg, ddepth, kernel=gk_theta)



cv.imshow("current Image", R_Theta)




cv.waitKey(0)
cv.destroyAllWindows()
