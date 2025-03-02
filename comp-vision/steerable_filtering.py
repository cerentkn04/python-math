import numpy as np
from PIL import Image
import cv2 as cv2

image = cv2.imread("sample.jpg")
image = np.array(image)
cv2.imshow("img",image)
ddepth = -12



Gblur_img=cv2.GaussianBlur(image,(5,5),0)
cv2.imshow(" gblur img",Gblur_img)

sigma = 1.5 # Standard deviation
G = lambda x, y: np.exp(-(x**2 + y**2) / (2 * sigma**2))
G0=lambda x, y: -2*x*G(x,y)
G90=lambda x, y: -2*y*G(x,y)

ksize = 5  
center = ksize // 2
x, y = np.meshgrid(np.arange(-center, center+1), np.arange(-center, center+1))



gkernel= G(x, y)
gk0 = G0(x, y)
gk90 = G90(x, y)

gk_theta = np.cos(0)*G0(x,y) + np.sin(0)*G90(x,y)
gk_theta /= np.sum(np.abs(gk_theta))

R_theta = cv2.filter2D(Gblur_img, ddepth, kernel=gk_theta) 


cv2.imshow("Filtered Image", R_theta)




cv2.waitKey(0)
cv2.destroyAllWindows()

