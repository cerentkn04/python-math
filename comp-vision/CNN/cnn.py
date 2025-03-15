import torch
import torch.nn as nn
from torchvision import transforms
import torchvision
import torch.optim as optim
import pandas as pd
import os
import cv2
from torch.utils.data import Dataset, DataLoader

# Dataset Class
class Fer2013Dataset(Dataset):
    def __init__(self, path, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform
        self.classes = os.listdir(path)

        for emotion_class in self.classes:
            class_index = self.classes.index(emotion_class)
            folder_path = os.path.join(path, emotion_class)

            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Grayscale images

                if img is not None:
                    img = cv2.resize(img, (48, 48))  # Resize all images to 48x48
                    self.data.append(img)
                    self.labels.append(class_index)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# Transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),  # Ensure single-channel image
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load Dataset
batch_size = 32
train_dataset = Fer2013Dataset('./arch/train', transform=transform)
test_dataset = Fer2013Dataset('./arch/test', transform=transform)

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model
class Fer2013(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=1, padding=1)  # 1 input channel for grayscale
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, stride=2)  # Reduce feature map size
        self.flatten = nn.Flatten()
        self.fc4 = nn.Linear(32 * 24 * 24, 10)  # Adjusted FC layer input size

    def forward(self, dataset):
        dataset = self.act1(self.conv1(dataset))
        dataset = self.drop1(dataset)
        dataset = self.act2(self.conv2(dataset))
        dataset = self.pool1(dataset)
        dataset = self.flatten(dataset)
        dataset = self.fc4(dataset)
        return dataset


# Model Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Fer2013().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
n_epochs = 20

for epoch in range(n_epochs):
    model.train()
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        y_pred = model(inputs)
        loss = loss_fn(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    acc = 0
    count = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            y_pred = model(inputs)
            acc += (torch.argmax(y_pred, 1) == labels).float().sum()
            count += len(labels)

    acc /= count
    print(f"Epoch {epoch + 1}: Model Accuracy {acc * 100:.2f}%")
