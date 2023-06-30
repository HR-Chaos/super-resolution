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
from scipy.spatial.distance import cosine

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

    # Use bicubic interpolation to upscale the image
    upscaled_image = image.resize((width*scale_factor, height*scale_factor), Image.BICUBIC)

    # Return the upscaled image
    return upscaled_image

def cosine_similarity(image1, image2):
    #image1 & 2 are numpy nd arrays
    
    flat_array1 = image1.flatten()
    flat_array2 = image2.flatten()
    
    # Normalize feature vectors
    normalized_array1 = flat_array1 / 255.0
    normalized_array2 = flat_array2 / 255.0

    # Calculate cosine similarity
    similarity = 1 - cosine(normalized_array1, normalized_array2)

    return similarity
    