'''
We're gonna be implementing LeNet (1998) by Yann LeCun.
For predicting digits from images of handwritten digits.
'''



# ruff: noqa: E402 # to ignore "imports not on top of the file" warning

import torch
import torch.nn as nn

# nn.Module - the base class for all neural networks in PyTorch
class LeNet(nn.Module):

    # ARCHITECTURE
    def __init__(self):
        super().__init__()
        # define layers
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        pass

    # HOW DATA FLOWS
    def forward(self, image):

        # define forward pass
        conv1_out = self.conv1(image)
        tanh1_out = torch.tanh(conv1_out) # activation function, tanh is used in the original LeNet paper
        pool1_out = self.pool1(tanh1_out)
        conv2_out = self.conv2(pool1_out)
        tanh2_out = torch.tanh(conv2_out)
        pool2_out = self.pool2(tanh2_out)
        flatten_out = pool2_out.view(-1, 256) # Flatten and feed to fully connected layers. -1 - infer batch size, 256 - no.of feature after flattening
        fc1_out = self.fc1(flatten_out)
        tanh3_out = torch.tanh(fc1_out)
        fc2_out = self.fc2(tanh3_out)
        tanh4_out = torch.tanh(fc2_out)
        fc3_out = self.fc3(tanh4_out) # output layer - raw scores for each digit (0-9). Therefore no activation - let them stay as is.
        
        return fc3_out


from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Now we need 3 things to train - Dataset, loss function, and optimizer.

# DATASET
# Load MNIST
transform = transforms.ToTensor() # converts images to PyTorch tensors and scales pixels from [0, 255] to [0, 1]
train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform) # download the MNIST dataset, and apply the transform to each image
test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True) # feed data in batches of 64, and shuffle the data for training
test_loader = DataLoader(test_data, batch_size=64)



import torch.optim as optim

if __name__ == "__main__":

    # In PyTorch, model training is external to the model.
    # The model only defines the Architecture and the data flow (forward pass).

    model = LeNet()
    print(model)

    # LOSS FUNCTION
    loss_fn = nn.CrossEntropyLoss() # loss function for multi-class classification, we're doing 0-9 digit classification, so we have 10 classes.
    # OPTIMZER
    optimizer = optim.SGD(model.parameters(), lr=0.01) # Stochastic Gradient Descent optimizer, with learning rate of 0.01. This will be used to update the weights.

    # train for 5 epochs
    for epoch in range(5):
        # Get a batch of data
        for images, targets in train_loader:
            optimizer.zero_grad() # zero the gradients before each batch, otherwise they will accumulate
            outputs = model(images) # forward pass
            loss = loss_fn(outputs, targets)
            loss.backward() # backpropagation - PyTorch does it automatically - computes the gradients for all weights in the model
            optimizer.step() # update the weights using the computed gradients
        print(f"Epoch: {epoch}, Loss: {loss.item()}") # print the loss after each epoch
    
    # TESTING
    total, correct = 0, 0
    with torch.no_grad(): # tells PyTorch I'm just testing no need to compute gradients
        for images, targets in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1) # gives the highest prob and it's index (predicted class/digit). A row in outputs is raw scores for each digit, for each image, in a batch.
            total += targets.size(0) # basically the no.of images in the batch.
            correct += (predicted == targets).sum().item() # (predicted == targets) gives a boolean tensor, where True means correct prediction. We sum them to get the total number of correct predictions, and convert to item to get the scalar value.
    print(f"Test Accuracy: {correct / total:.4f}")


# So essentialy we have this:
# Define model - Architecture and forward
# Train model - dataset, loss function (calculate loss, backpropagates loss), optimiser (update weights)