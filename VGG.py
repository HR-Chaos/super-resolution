import torch
from torch import nn
from torchvision.models import vgg19, VGG19_Weights

class VGGLoss(nn.Module):
    def __init__(self, device):
        super(VGGLoss, self).__init__()
        model = vgg19(weights = VGG19_Weights.DEFAULT)
        features = model.features
        self.features = features.to(device)
        self.loss_arr = []
        # Use the layers up to the second max pooling layer for the loss calculation
        self.layer_name_mapping = {
            '1': "relu1_1", 
            '3': "relu1_2", 
            '6': "relu2_1", 
            '8': "relu2_2", 
            '11': "relu3_1", 
            '13': "relu3_2", 
            '15': "relu3_3", 
            '17': "relu3_4", 
            '20': "relu4_1", 
            '22': "relu4_2", 
            '24': "relu4_3", 
            '26': "relu4_4", 
            '29': "relu5_1",
            '31': "relu5_2",
            '33': "relu5_3",
            '35': "relu5_4"
        }

    def forward(self, input, target):
        self.loss_arr = []
        input_features = self.get_features(input)
        target_features = self.get_features(target)
        loss = 0
        for input_feature, target_feature in zip(input_features, target_features):
            l = nn.functional.mse_loss(input_feature, target_feature)
            loss += l
            self.loss_arr.append(l.item())

        # print(loss_arr)
        return loss, self.loss_arr
    
    def get_features(self, x):
        features = []
        for name, layer in self.features._modules.items():
            x = layer(x)
            # print(layer)
            if name in self.layer_name_mapping:
                features.append(x)
        return features