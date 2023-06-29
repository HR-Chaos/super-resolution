import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1):
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
    
    
class NeuralNet2(nn.Module):
    '''
    Neural Network for Super Resolution
    scale_factor: upscaling factor for one iteration
    num_scale: number of upscaling iterations
    block_size: size of the block for the residual block
    '''
    def __init__(self, scale_factor=4):
        super(NeuralNet2, self).__init__()
        self.scale_factor = scale_factor

        # first convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # first residual block
        self.residual1 = self.make_layer(ResidualBlock, 3, 256)

        # second convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        # second residual block
        self.residual2 = self.make_layer(ResidualBlock, 3, 512)

        self.upscale = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.PixelShuffle(self.scale_factor),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def make_layer(self, block, num_of_layer, channel_size=512):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(in_channels=channel_size, out_channels=channel_size, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        
        out = self.conv1(x)
        residual = out
        out = self.residual1(out)
        out = out + residual
        out = F.relu(out)
        
        out = self.conv2(out)
        residual = out
        out = self.residual2(out)
        out = out + residual
        out = F.relu(out)
        
        out = self.upscale(out)
        out = (out + 1) / 2  # Rescale from [-1,1] to [0,1]
        return out