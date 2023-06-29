'''
Neural Network with 4 seperate convolutions that
are concatenated and then upscaled by 4
'''

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_size, out_channels=in_size, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.PReLU()
        )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual
        return F.relu(out)

class NeuralNet3(nn.Module):
    def __init__(self):
        super(NeuralNet3, self).__init__()

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )

        self.residual1 = self.make_layer(ResidualBlock, 3, 96)

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, stride=1, padding=2),
            nn.PReLU()
        )

        self.residual2 = self.make_layer(ResidualBlock, 3, 48)

        self.conv3_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        self.residual3 = self.make_layer(ResidualBlock, 3, 24)

        self.conv4_1 = nn.Sequenctial(
            nn.Conv2d(in_channels=15, out_channels=24, kernel_size=15, stride=1, padding=7),
            nn.PReLU()
        )

        self.residual4 = self.make_layer(ResidualBlock, 3, 24)

        self.resid = self.make_layer(ResidualBlock, 3, 192)

        self.upscale = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=768, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.PixelShuffle(4),
            nn.Conv2d(in_channels=48, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def make_layer(self, block, num_of_layer, channel_size=512):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(in_channels=channel_size, out_channels=channel_size, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv1_1(x)
        residual = x1
        x1 = self.residual1(x1)
        x1 = x1 + residual
        x1 = F.prelu(x1)

        x2 = self.conv2_1(x)
        residual = x2
        x2 = self.residual2(x1)
        x2 = x2 + residual

        x3 = self.conv3_1(x)
        residual = x3
        x3 = self.residual3(x1)
        x3 = x3 + residual

        x4 = self.conv4_1(x)
        residual = x4
        x4 = self.residual4(x1)
        x4 = x4 + residual

        out = torch.cat((x1, x2, x3, x4), 1)
        residual = out
        out = self.resid(out)
        out = out + residual
        out = F.prelu(out)
        
        out = self.upscale(out)
        out = (out + 1) / 2  # Rescale from [-1,1] to [0,1]
        return out