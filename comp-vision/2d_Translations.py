import numpy as np
from PIL import Image

# Load image
image = Image.open("sample2.jpg")
width, height = image.size

# Define the rotation angle in degrees and scaling factor
theta = 20  # Rotation angle in degrees
s = 1  # Scaling factor
tx, ty = 50, 30  # Translation vector

# Convert angle to radians
theta_rad = np.radians(theta)

# Define the 2D rotation matrix for the angle
R = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
              [np.sin(theta_rad), np.cos(theta_rad)]])

# Create the homogeneous transformation matrix (scaled rotation + translation)
transformation_matrix = np.array([[s * R[0, 0], s * R[0, 1], tx],
                                   [s * R[1, 0], s * R[1, 1], ty],
                                   [0, 0, 1]])

# Generate a grid of (x, y) coordinates
y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
ones = np.ones_like(x_coords)  # Extra row for homogeneous coordinates
coords = np.stack([x_coords, y_coords, ones])  # Shape (3, height * width)

# Apply the transformation
transformed_coords = transformation_matrix @ coords.reshape(3, -1)

# Extract new (x, y) coordinates
x_transformed = transformed_coords[0].reshape(height, width).astype(int)
y_transformed = transformed_coords[1].reshape(height, width).astype(int)

# Create an empty output image
transformed_image = Image.new("RGB", (width, height))

# Map pixels to new locations (without interpolation)
for i in range(height):
    for j in range(width):
        new_x, new_y = x_transformed[i, j], y_transformed[i, j]
        if 0 <= new_x < width and 0 <= new_y < height:  # Keep inside bounds
            transformed_image.putpixel((new_x, new_y), image.getpixel((j, i)))

# Show the transformed image
transformed_image.show()
