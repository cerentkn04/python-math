import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
import os
import cv2
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



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
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  

                if img is not None:
                    img = cv2.resize(img, (48, 48))  
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


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])





batch_size = 32
train_dataset = Fer2013Dataset('./arch/train', transform=transform)
test_dataset = Fer2013Dataset('./arch/test', transform=transform)

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class Fer2013(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2) 
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2)  
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, stride=2)  
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 6 * 6, 512) 
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 7)  

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)
        
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = self.fc2(x)
        return x


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
    all_preds = []
    all_labels = []
    acc = 0
    count = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            y_pred = model(inputs)
            predictions = torch.argmax(y_pred, dim=1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            acc += (torch.argmax(y_pred, 1) == labels).float().sum()
            count += len(labels)

    acc /= count
    print(f"Epoch {epoch + 1}: Model Accuracy {acc * 100:.2f}%")


cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.classes)


plt.figure(figsize=(8, 8))
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix")
plt.show()