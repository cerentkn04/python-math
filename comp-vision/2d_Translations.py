import numpy as np
from PIL import Image

# Load image
image = Image.open("sample.jpg")
width, height = image.size


tx, ty = 0, 0
kx,ky= 0.2,0.1


T = np.array([[1, kx, tx], 
              [ky, 1, ty], 
              [0, 0, 1]])


y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
ones = np.ones_like(x_coords)  
coords = np.stack([x_coords, y_coords, ones]) 


transformed_coords = T @ coords.reshape(3, -1)


x_transformed = transformed_coords[0].reshape(height, width).astype(int)
y_transformed = transformed_coords[1].reshape(height, width).astype(int)


transformed_image = Image.new("RGB", (width, height))


for i in range(height):
    for j in range(width):
        new_x, new_y = x_transformed[i, j], y_transformed[i, j]
        if 0 <= new_x < width and 0 <= new_y < height:  
            transformed_image.putpixel((new_x, new_y), image.getpixel((j, i)))

transformed_image.show()
