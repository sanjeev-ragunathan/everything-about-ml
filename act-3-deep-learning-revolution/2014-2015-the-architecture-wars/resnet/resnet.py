'''
Building ResNet.
Transfer Learning.

ResNet-18.
We'll swap out the last layer and train it on CIFAR-10.
'''

# ruff: noqa: E402 # to ignore "imports not on top of the file" warning

from torchvision import models
import torch.nn as nn

class ResNet18(nn.Module):

    # ARCHITECTURE
    def __init__(self):
        super().__init__()
        
        # Load Pre-trained Model
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) # pretrained on ImageNet
        
        # Swap out the last layer
        # model.fc - the last layer of ResNet-18 it's a fully connected layer with 512 input features and 1000 output features (for the 1000 classes in ImageNet).
        # Replaced with our new fc layer for CIFAR-10.
        model.fc = nn.Linear(model.fc.in_features, 10)

        self.model = model
    
    # DATA FLOW
    def forward(self, images):
        return self.model(images)



# DATASET

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# how the data is transformed
# data augmentation and data normalization
train_transform = transforms.Compose([
    transforms.Resize(224), # ResNet-18 was trained on 224x224 images, so we need to resize CIFAR-10 images from 32x32 to 224x224
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Load CIFAR-10 dataset
train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
test_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

# decide how data is fed to the model
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)



import torch
import torch.optim as optim

if __name__ == "__main__":

    model = ResNet18()
    print(model)

    # LOSS FUNCTION - calculate error and backpropagates error
    loss_fn = nn.CrossEntropyLoss()
    # OPTIMIZER - updates weights
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # I have a Macbook M1
    # trying to access GPU on M1 macbook, if available. If not, use CPU.
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    # TRAIN LOOP
    for epoch in range(5):
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            # zero gradient
            optimizer.zero_grad()
            # forward
            outputs = model(images)
            # calculate loss
            loss = loss_fn(outputs, targets)
            # propagate loss
            loss.backward()
            # update weights
            optimizer.step()
        # print epoch - loss
        print(f"Epoch: {epoch}, Loss: {loss}")
    
    # TEST
    total, correct = 0, 0
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        print(f"Accuracy: {correct / total:.4f}")
