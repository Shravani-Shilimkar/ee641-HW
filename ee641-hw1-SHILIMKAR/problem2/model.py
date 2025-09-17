# problem2/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """Shared encoder for both network architectures."""
    def __init__(self):
        super().__init__()
        # Conv1: 128 -> 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Conv2: 64 -> 32
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Conv3: 32 -> 16
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Conv4: 16 -> 8
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        return c2, c3, c4 # Return intermediate layers for skip connections

class HeatmapNet(nn.Module):
    """Network for heatmap-based keypoint regression."""
    def __init__(self, num_keypoints=5):
        super().__init__()
        self.encoder = Encoder()
        
        # Deconv4: 8 -> 16
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128), nn.ReLU()
        )
        # Deconv3: 16 -> 32
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2), # 128 (from deconv4) + 128 (from encoder c3) = 256
            nn.BatchNorm2d(64), nn.ReLU()
        )
        # Deconv2: 32 -> 64
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2), # 64 (from deconv3) + 64 (from encoder c2) = 128
            nn.BatchNorm2d(32), nn.ReLU()
        )
        
        # Final output layer
        self.final_conv = nn.Conv2d(32, num_keypoints, kernel_size=1)

    def forward(self, x):
        # Encoder path
        c2, c3, c4 = self.encoder(x)
        
        # Decoder path with skip connections
        d4 = self.deconv4(c4)
        d4_cat = torch.cat([d4, c3], dim=1)
        
        d3 = self.deconv3(d4_cat)
        d3_cat = torch.cat([d3, c2], dim=1)
        
        d2 = self.deconv2(d3_cat)
        
        # Final layer
        output = self.final_conv(d2)
        return output

class RegressionNet(nn.Module):
    """Network for direct coordinate regression."""
    def __init__(self, num_keypoints=5):
        super().__init__()
        self.encoder = Encoder()
        
        # Regression Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, num_keypoints * 2),
            nn.Sigmoid() # To ensure output is in [0, 1]
        )

    def forward(self, x):
        _, _, features = self.encoder(x)
        coords = self.head(features)
        return coords