import numpy as np
import matplotlib.pyplot as plt
import cv2
from neuralNet1 import NeuralNet1
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
from customDataset import CustomDataset
from torch.utils.data import DataLoader
from scipy.ndimage import convolve
from FourierLoss import BandFilterLoss, BandFilterLossTorch, FourierHeatMap,FourierLossTorch
from scipy.stats import multivariate_normal

def upscale_image_bicubic(image, scale_factor):
    if isinstance(image, str):
        # Load image from path
        image = Image.open(image)
    elif isinstance(image, torch.Tensor):
        # Convert Torch tensor to PIL image
        image = image.cpu().numpy().squeeze().transpose((1, 2, 0))
        image = Image.fromarray((image * 255).astype('uint8'))
    elif isinstance(image, Image.Image):
        # Make a copy of the PIL image
        image = image.copy()
    else:
        raise ValueError("Unsupported input type")

    # Calculate the new dimensions
    width, height = image.size
    new_width = width * scale_factor
    new_height = height * scale_factor

    # Use bicubic interpolation to upscale the image
    upscaled_image = image.resize((new_width, new_height), Image.BICUBIC)

    # Return the upscaled image
    return upscaled_image