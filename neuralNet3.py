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
            nn.BatchNorm2d(out_size),
            nn.PReLU()
        )
        
        self.final_prelu = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual
        return self.final_prelu(out)

class NeuralNet3(nn.Module):
    def __init__(self):
        super(NeuralNet3, self).__init__()

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )

        self.residual1_1 = self.make_layer(ResidualBlock, 96, 96)

        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, stride=1, padding=2),
            nn.PReLU()
        )

        self.residual1_2 = self.make_layer(ResidualBlock, 48, 48)

        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        self.residual1_3 = self.make_layer(ResidualBlock, 24, 24)

        self.conv1_4 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=15, stride=1, padding=7),
            nn.PReLU()
        )

        self.residual1_4 = self.make_layer(ResidualBlock, 24, 24)

        self.resid = self.make_layer(ResidualBlock, 192, 192)

        self.upscale = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=768, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.PixelShuffle(4),
            nn.Conv2d(in_channels=48, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        self.final_prelu = nn.PReLU()

    def make_layer(self, block, in_size, out_size):
        layers = []
        for _ in range(3):
            layers.append(block(in_size, out_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv1_1(x)
        residual = x1
        x1 = self.residual1_1(x1)
        x1 = x1 + residual
        x1 = self.final_prelu(x1)

        x2 = self.conv1_2(x)
        residual = x2
        x2 = self.residual1_2(x2)
        x2 = x2 + residual
        x2 = self.final_prelu(x2)

        x3 = self.conv1_3(x)
        residual = x3
        x3 = self.residual1_3(x3)
        x3 = x3 + residual
        x3 = self.final_prelu(x3)

        x4 = self.conv1_4(x)
        residual = x4
        x4 = self.residual1_4(x4)
        x4 = x4 + residual
        x4 = self.final_prelu(x4)

        out = torch.cat((x1, x2, x3, x4), 1)    # concat on dim=1 because batches are in dim=0
        residual = out
        out = self.resid(out)
        out = out + residual
        out = self.final_prelu(out)

        out = self.upscale(out)
        out = (out + 1) / 2  # Rescale from [-1,1] to [0,1]
        return out