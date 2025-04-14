import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


    
class Generator(nn.Module):
    def __init__(self, noise_dim, n_class):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(noise_dim, 140 * 64)
        self.bn1 = nn.BatchNorm1d(140 * 64, momentum=0.9)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1)
        #Hout​=(Hin​−1)×stride−2×padding+kernel_size+output_padding # 20
        #𝑊𝑜𝑢𝑡=(𝑊𝑖𝑛−1)×stride−2×padding+kernel_size+output_padding  # 28
        self.bn2 = nn.BatchNorm2d(32, momentum=0.9)
        self.deconv2 = nn.ConvTranspose2d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=1)
        #Hout​=(Hin​−1)×stride−2×padding+kernel_size+output_padding # 40
        #𝑊𝑜𝑢𝑡=(𝑊𝑖𝑛−1)×stride−2×padding+kernel_size+output_padding  # 56
        self.fc2 = nn.Linear(40 * 56, 20 * 20)
        self.fc3 = nn.Linear(20 * 20, n_class)


    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = x.view(-1, 64, 10, 14)  # Reshape to match TensorFlow's behavior
        x = F.relu(self.bn2(self.deconv1(x)))
        x = self.deconv2(x)
        logits = x.view(-1, 40 * 56)  # Flatten
        outputs = torch.tanh(logits)
        mid = torch.sigmoid(self.fc2(logits))
        pred = self.fc3(mid)
        return logits, outputs, pred
    
class Discriminator(nn.Module):
    def __init__(self, n_class):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding="same")
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64* 20* 28 + 40, 1)
        self.fc2 = nn.Linear(64* 20* 28 + 40, n_class)

    def forward(self, x, cond):
        x = x.view(-1, 1, 40, 56)
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = x.view(-1, 64* 20* 28)
        x = torch.cat((x, cond[: ,20:]), 1)

        logits = self.fc1(x)
        outputs_2 = self.fc2(x)
        return logits, logits, outputs_2
