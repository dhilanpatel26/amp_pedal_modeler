import torch
import torch.nn as nn
import torch.nn.functional as F


class GuitarAmpSimulator(nn.Module):
    def __init__(self, input_length=1440):
        super(GuitarAmpSimulator, self).__init__()
        # defining layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        # dynamically sizing fully connected layers to match input length
        self.fc1 = nn.Linear(128 * (input_length // 8), 512)
        self.fc2 = nn.Linear(512, input_length)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # apply convolutional layers with ReLU activation and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten tensor
        x = self.flatten(x)
        # dropout to prevent overfitting
        x = self.dropout(x)
        # apply fully connected layers with ReLU activation
        x  = F.relu(self.fc1(x))
        # output layer, linear activation
        x = self.fc2(x)
        # reshape tensor
        x = x.unsqueeze(0)
        return x
    