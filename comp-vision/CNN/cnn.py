import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import cv2 as cv 
from torchvision import transforms
import matplotlib.pyplot as plt


image=Image.open("../sample2.jpg")

transform = transforms.Compose([
    transforms.ToTensor()
])

image = transform(image).unsqueeze(0)
image = image
conv1 = nn.Conv2d(3, 64, kernel_size=(3,3), stride=1, padding=1)
act1 = nn.ReLU()
drop1 = nn.Dropout(0.3)
 
conv2 = nn.Conv2d(64, 32, kernel_size=(3,3), stride=1, padding=1)
act2 = nn.ReLU()
pool2 = nn.MaxPool2d(kernel_size=(2, 2))
flatten = nn.Flatten()

P1 = nn.MaxPool2d(3, stride=2)
imm= act1(conv1(image)) 
imm= drop1(imm)
img= act2(conv2(imm))
img =pool2(img)
flatten_filters= flatten(img)
linear= nn.Linear(flatten_filters.shape[1], 30)
linear_output= linear(flatten_filters)

print("FLATTT", linear_output)

num_filters = linear_output.shape[1] 
rows, cols = 2, 4  

fig, axes = plt.subplots(rows, cols, figsize=(15, 8))
fig.suptitle("Feature Maps from First Convolutional Layer")

for i in range(rows * cols):
    ax = axes[i // cols, i % cols]
    ax.imshow(imm[0, i].detach().numpy(), cmap='gray')
    ax.axis("off")
    ax.set_title(f"Filter {i}")

plt.tight_layout()
plt.show()
