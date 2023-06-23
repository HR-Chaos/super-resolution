import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self,in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.PreLU()
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
    def __init__(self, scale_factor=4, num_scale=1, block_size=4):
        super(NeuralNet, self).__init__()
        self.scale_factor = scale_factor
        self.num_scale = num_scale
        self.block_size = block_size

        # 2 convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(48)
        )

        # residual layers
        self.residual = self.make_layer(ResidualBlock, self.block_size)

        # upscaling layer
        self.upscale = nn.Sequential(
            nn.PixelShuffle(self.scale_factor),
            nn.Tanh()
        )
    
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # upscale 1
        out = self.conv(x)
        residual = out          # residual block
        out = self.residual(out)
        out += residual
        out = self.upscale(out)
        return out