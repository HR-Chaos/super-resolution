import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual
        return F.relu(out)


class NeuralNet1(nn.Module):
    '''
    Neural Network for Super Resolution
    scale_factor: upscaling factor for one iteration
    num_scale: number of upscaling iterations
    block_size: size of the block for the residual block
    '''
    def __init__(self, scale_factor=4, num_scale=1, block_size=4):
        super(NeuralNet1, self).__init__()
        self.scale_factor = scale_factor
        self.num_scale = num_scale
        self.block_size = block_size

        # first convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU()
            
        )

        # residual layers
        self.residual1 = self.make_layer(ResidualBlock, self.block_size)


        self.upscale1 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.PixelShuffle(self.scale_factor),
            nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # upscale 1
        out = self.conv1(x)
        residual = out
        out = self.residual1(out)
        out = out + residual
        out = F.relu(out)
        out = self.upscale1(out)
        out = (out + 1) / 2  # Rescale from [-1,1] to [0,1]
        return out