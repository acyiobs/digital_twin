"""
author:Shuaifeng
data:10/8/2022
"""
import torch
import numpy as np
import torch.nn as nn
    
class FullyConnected(nn.Module):
    def __init__(self, num_classes, hidden_size=256):
        super(FullyConnected, self).__init__()
        self.fc1 = nn.Linear(4, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.5)
        self.name = 'FullyConnected'

    def forward(self, x):
        y = self.fc1(x)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.relu(y)
        y = self.fc3(y)
        y = self.relu(y)
        y = self.classifier(y)
        return y

class FullyConnected2(nn.Module):
    def __init__(self, num_classes, hidden_size=64):
        super(FullyConnected2, self).__init__()
        self.fc1 = nn.Linear(4, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)

        self.classifier = nn.Linear(hidden_size, num_classes)

        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.5)
        self.name = 'FullyConnected'

    def forward(self, x):
        y = self.fc1(x)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.relu(y)
        y = self.fc3(y)
        y = self.relu(y)
        y = self.fc4(y)
        y = self.relu(y)
        y = self.classifier(y)
        return y

class FullyConnected3(nn.Module):
    def __init__(self, num_classes, hidden_size=256):
        super(FullyConnected3, self).__init__()
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.classifier = nn.Linear(64, num_classes)

        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.5)
        self.name = 'FullyConnected'

    def forward(self, x):
        y = self.fc1(x)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.relu(y)
        y = self.fc3(y)
        y = self.relu(y)
        y = self.fc4(y)
        y = self.relu(y)
        y = self.fc5(y)
        y = self.relu(y)
        y = self.classifier(y)
        return y
