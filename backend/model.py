import torch
import torch.nn as nn


class FinancialSpectrogramCNN(nn.Module):

    def __init__(self, num_features):
        super(FinancialSpectrogramCNN, self).__init__()
        
        # Input channels = num_features (e.g., 5 for Open, High, Low, Close, Volume)
        self.conv1 = nn.Conv2d(in_channels=num_features, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Flatten and feed into a fully connected layer for regression (predicting continuous price)
        self.flatten = nn.Flatten()
        
        # Note: The input features to the Linear layer depend on your STFT window size.
        # You will need to calculate or dynamically size this based on your specific T x F dimensions.
        # We will use a placeholder 'dummy' forward pass in main.py to find this exact number.
        self.fc1 = nn.LazyLinear(out_features=64) 
        self.relu3 = nn.ReLU()
        
        # Output layer: Predicting 1 value (e.g., the next day's Close price)
        self.fc_out = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        # x shape: (Batch, Channels, Frequencies, Time_Windows)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        out = self.fc_out(x)
        return out
