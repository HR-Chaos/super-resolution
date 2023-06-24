import torch
from torch import nn
from torchvision.models import vgg19, VGG19_Weights

class VGGLoss(nn.Module):
    def __init__(self, device):
        super(VGGLoss, self).__init__()
        model = vgg19(weights = VGG19_Weights.DEFAULT)
        features = model.features
        self.features = features.to(device)
        # Use the layers up to the second max pooling layer for the loss calculation
        self.layer_name_mapping = {
            '3': "relu1_2", 
            '13': "relu3_2",
            '15': "relu3_3",
            '17': "relu3_3", 
            '20': "relu4_1",
        }

    def forward(self, input, target):
        input_features = self.get_features(input)
        target_features = self.get_features(target)
        loss = 0
        for input_feature, target_feature in zip(input_features, target_features):
            loss += nn.functional.mse_loss(input_feature, target_feature)
        return (1/len(self.layer_name_mapping))*loss/2

    def get_features(self, x):
        features = []
        for name, layer in self.features._modules.items():
            x = layer(x)
            if name in self.layer_name_mapping:
                features.append(x)
        return features