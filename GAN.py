import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64):
        super(Generator, self).__init__()

        self.nz = nz
        self.ngf = ngf

        self.fc = nn.Sequential(
            # Input is Z, going into a convolution
            nn.Linear(nz, ngf * 8 * 64 * 64, bias=False),
            nn.InstanceNorm1d(ngf * 8 * 64 * 64),
            nn.ReLU(True),
            # The output of this layer is reshaped before being fed into the convolutional layers
        )

        # initial convolution block
        # takes 3 channels to 64 channels
        self.conv1 = nn.Sequential(
            # initial convolution block - input is (ngf*8) x 64 x 64
            nn.Conv2d(ngf * 8, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
        )

        # upscaling block. Increases resolution by 2 times
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )

        # final convolution block to get 3 channels
        self.conv5 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        out = self.fc(x)
        out = out.view(out.size(0), self.ngf * 8, 64, 64)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        return out
    

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            # can add more layers here

            nn.AdaptiveAvgPool2d(1), 
            nn.Conv2d(64, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)