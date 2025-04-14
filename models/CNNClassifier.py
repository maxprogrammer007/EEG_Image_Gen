import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
  

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding="same")
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding="same")
        self.fc1 = nn.Linear(64 * 2 * 3, 40) # Adjust based on the input dimensions after pooling
        self.fc2 = nn.Linear(40, 5) # Number of classes is 5

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 2 * 3) # Adjust based on the input dimensions after pooling
        x = F.dropout(x, 0.5)
        x1 = torch.sigmoid(self.fc1(x))
        x2 = self.fc2(x1)
        return x2 , x1