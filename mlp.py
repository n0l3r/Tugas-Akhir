import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import tqdm

class MLP(nn.Module):
    def __init__(self, input_size=320*320*3, hidden_size=[128, 64], num_classes=3):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size[0])  # hidden_size[0] adalah ukuran hidden layer pertama
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])  # hidden_size[1] adalah ukuran hidden layer kedua
        self.fc3 = nn.Linear(hidden_size[1], num_classes)
    
    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x