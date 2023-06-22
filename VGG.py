import torch
from torch import nn
from torchvision.models import vgg19

class VGGLoss(nn.Module):
    def __init__(self, device):
        super(VGGLoss, self).__init__()
        model = vgg19(pretrained=True)
        features = model.features
        self.features = features.to(device)
        # Use the layers up to the second max pooling layer for the loss calculation
        self.layer_name_mapping = {
            '3': "relu1_2", 
            '8': "relu2_2", 
            '17': "relu3_3", 
            '26': "relu4_3", 
            '35': "relu5_3"
        }

    def forward(self, input, target):
        input_features = self.get_features(input)
        target_features = self.get_features(target)
        loss = 0
        for input_feature, target_feature in zip(input_features, target_features):
            loss += nn.functional.mse_loss(input_feature, target_feature)
        return loss

    def get_features(self, x):
        features = []
        for name, layer in self.features._modules.items():
            x = layer(x)
            if name in self.layer_name_mapping:
                features.append(x)
        return features