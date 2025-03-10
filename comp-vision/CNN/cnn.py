import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


image=Image.open("../sample.jpg")

print(image)
transform = transforms.Compose([transforms.PILToTensor()])


tensor_ = transform(image)

CL1=nn.Conv2d(6, 33, 3,stride=2, padding=1)
