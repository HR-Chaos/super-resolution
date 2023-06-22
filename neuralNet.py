import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self,in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return out


class NeuralNet(nn.Module):
    '''
    Neural Network for Super Resolution
    scale_factor: upscaling factor for one iteration
    num_scale: number of upscaling iterations
    block_size: size of the block for the residual block
    '''
    def __init__(self, scale_factor=2, num_scale=2, block_size=4):
        super(NeuralNet, self).__init__()
        self.scale_factor = scale_factor
        self.num_scale = num_scale
        self.block_size = block_size

        # first convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(128)
        )

        # residual layers
        self.residual2 = self.make_layer(ResidualBlock, self.block_size)

        # upscaling layer
        self.upscale1 = nn.Sequential(
            nn.PixelShuffle(self.scale_factor),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=9, stride=1, padding=4)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(128)
        )

        self.residual2 = self.make_layer(ResidualBlock, self.block_size)

        self.upscale2 = nn.Sequential(
            nn.PixelShuffle(self.scale_factor),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=9, stride=1, padding=4), 
            nn.Tanh()
        )
    
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # upscale 1
        out = self.conv1(x)
        residual = out          # residual block
        out = self.residual2(out)
        out += residual
        out = self.upscale1(out)

        # upscale 2
        out = self.conv2(out)
        residual = out          # residual block    
        out = self.residual2(out)
        out += residual
        out = self.upscale2(out)
        return out