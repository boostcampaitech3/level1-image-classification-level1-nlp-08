import torch
import torch.nn as nn
import timm

# ConvNext Base : 224x224
class convnext(nn.Module):
    def __init__(self, num_classes):
        super(convnext, self).__init__()
        self.convnext = timm.create_model('convnext_base', pretrained=True, num_classes = num_classes)
        
    def forward(self, x):
        return self.convnext(x)


# EfficientNet B6 : 528x528
# b0 : 224x224
class effnet(nn.Module):
    def __init__(self, num_classes):
        super(effnet, self).__init__()
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.efficientnet(x)


# ResNext101 32x8d : 224x224
class resnext(nn.Module):
    def __init__(self, num_classes):
        super(resnext, self).__init__()
        self.resnext = timm.create_model('resnext101_32x8d', pretrained=True, num_classes=num_classes, drop_rate = 0.5)

    def forward(self, x):
        return self.resnext(x)

