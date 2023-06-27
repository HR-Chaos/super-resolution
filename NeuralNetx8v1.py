import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # residual layers
        self.residual =  nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias = False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias = False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(1024, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        
    def make_layer(self, block, num_of_layer, channel_size):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(channel_size))
        return nn.Sequential(*layers)

    def forward(self, input):
        out = self.conv1(input)
        residual = out
        out = self.residual(out)
        out = out + residual
        out = self.conv2(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channel_size):
        super(ResidualBlock, self).__init__()
        self.channel_size = channel_size
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.channel_size, out_channels=self.channel_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.channel_size, out_channels=self.channel_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.channel_size),
            nn.ReLU()
        )
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual
        return F.relu(out)
    
class NeuralNet(nn.Module):
    def __init__(self):
        
        super(NeuralNet, self).__init__()
        
        # first convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()   
        )

        # residual layers
        self.residual1 = self.make_layer(ResidualBlock, 2, 128)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()   
        )
        
        # residual layers
        self.residual2 = self.make_layer(ResidualBlock, 2, 256)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()   
        )
        
        # residual layers
        self.residual3 = self.make_layer(ResidualBlock, 2, 512)

        self.upscale = nn.Sequential(
            nn.PixelShuffle(8),
            nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def make_layer(self, block, num_of_layer, channel_size):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(channel_size))
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
        
        out = self.conv3(out)
        
        residual = out
        out = self.residual3(out)
        out = out + residual
        out = F.relu(out)
        
        out = self.upscale(out)
        out = (out + 1) / 2  # Rescale from [-1,1] to [0,1]
        return out