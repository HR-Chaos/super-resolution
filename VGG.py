import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights

class VGGLoss(nn.Module):
    def __init__(self, device):
        super(VGGLoss, self).__init__()
        model = vgg19(weights=VGG19_Weights.DEFAULT)
        features = model.features
        self.features = features.to(device)
        # Use the layers up to the third max pooling layer for the loss calculation
        self.layer_name_mapping = {
            # '1': "relu1_1", 
            # '3': "relu1_2", 
            # '6': "relu2_1", 
            # '8': "relu2_2", 
            '11': "relu3_1", 
            '13': "relu3_2", 
            '15': "relu3_3", 
            '17': "relu3_4", 
            '20': "relu4_1", 
            '22': "relu4_2", 
            # '24': "relu4_3", 
            # '26': "relu4_4", 
            # '29': "relu5_1",
            # '31': "relu5_2",
            # '33': "relu5_3",
            # '35': "relu5_4"
        }

    def forward(self, input, target):
        loss=0
        for input_feature, target_feature in self.get_features(input,target):
            loss+=F.mse_loss(input_feature, target_feature).item()
        return loss
    
    def get_features(self,x,y):
        features=[]
        for name, layer in self.features._modules.items():
            x = layer(x)
            y = layer(y)
            if name in self.layer_name_mapping:
                features.append((x,y))
                if name=='22':
                    break
        return features
    # def forward(self,input,target):
    #     loss = 0
    #     for name,layer in self.features._modules.items():
    #         input = layer(input)
    #         target = layer(target)
    #         if name in self.layer_name_mapping:
    #             loss+=F.mse_loss(input,target).item()
    #             if name=='22':
    #                 break
    #     return loss