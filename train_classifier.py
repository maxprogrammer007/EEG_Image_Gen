import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from models.CNNClassifier import *
from utils import *
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

EPOCHS = 1000

model = CNNClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.0005)

data = np.load(r"data_preprocessing\data.npy")


n_classes = 5
feature = data[:, :140]
label = data[:, -1]


data_original = data
n_group = 5
batch_size = int(label.shape[0]/n_group)

train_fea = []

for i in range(n_group):
    f = feature[int(0+batch_size*i):int(batch_size+batch_size*i)]
    train_fea.append(f)

train_label = []
for i in range(n_group):
    f = label[int(0 + batch_size * i):int(batch_size + batch_size * i)]
    train_label.append(f)

test_fea = torch.tensor(train_fea[-1], dtype=torch.float32)
test_label = torch.tensor(train_label[-1], dtype=torch.long)

train_data = [(torch.tensor(train_fea[i], dtype=torch.float32), torch.tensor(train_label[i], dtype=torch.long)) for i in range(n_group-1)]

train_losses,test_losses = train(model,
                                  train_data=train_data,
                                  test_fea=test_fea,
                                  test_label=test_label,
                                  optimizer=optimizer,
                                  criterion=criterion,
                                  device="cuda",
                                  EPOCHS=EPOCHS)

torch.save(model.state_dict(), r"models/weights/full_model.pth")

plot_losses(train_losses=train_losses,
            test_losses=test_losses,
            epochs=EPOCHS,
            save_path="plots/")

extract_features(model=model,train_fea=train_fea,device=device,data_original=data_original)