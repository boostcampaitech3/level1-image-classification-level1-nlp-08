import torch
import torch.nn as nn
import timm
from fastai import *
from fastai.vision import *
from fastai.vision.learner import create_body, create_head
from fastai.layers import CrossEntropyFlat
from torchvision import transforms

# CoAtNet : 
class coatnet(nn.Module):
    def __init__(self, num_classes):
        super(coatnet, self).__init__()
        self.coatnet = timm.create_model('coat_mini', img_size=224, pretrained=True, num_classes = num_classes,drop_rate=0.5)

        # if freeze == True:
        #     timm.utils.freeze(self.resnext)
        #     self.convnext.fc.weight.requires_grad = True
        #     self.convnext.fc.bias.requires_grad = True

    def forward(self, x):
        return self.coatnet(x)


# ConvNext Base : 224x224
class convnext(nn.Module):
    def __init__(self, num_classes):
        super(convnext, self).__init__()

        self.convnext = timm.create_model('convnext_base', pretrained=True, num_classes = num_classes, drop_rate=0.5)

        # if freeze == True:
        #     timm.utils.freeze(self.resnext)
        #     self.convnext.fc.weight.requires_grad = True
        #     self.convnext.fc.bias.requires_grad = True

    def forward(self, x):
        return self.convnext(x)


# EfficientNet B6 : 528x528
# b0 : 224x224
class effnet(nn.Module):
    def __init__(self, num_classes):
        super(effnet, self).__init__()

        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)

        # if freeze == True:
        #     timm.utils.freeze(self.efficientnet)
        #     self.efficientnet.classifier.weight.requires_grad = True
        #     self.efficientnet.classifier.bias.requires_grad = True

    def forward(self, x):
        return self.efficientnet(x)


# ResNext101 32x8d : 224x224
class resnet(nn.Module):
    def __init__(self, num_classes):
        super(resnet, self).__init__()

        self.resnet = timm.create_model('resnext34', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.resnet(x)

class MultiTaskModel(nn.Module):
    """
    Creates a MTL model with the encoder from "arch" and with dropout multiplier ps.
    """
    def __init__(self, arch,ps=0.5):
        super(MultiTaskModel,self).__init__()
        self.encoder = create_body(arch)        #fastai function that creates an encoder given an architecture
        self.fc1 = create_head(1024,3,ps=ps)    #fastai function that creates a head
        self.fc2 = create_head(1024,2,ps=ps)
        self.fc3 = create_head(1024,3,ps=ps)

    def forward(self,x):

        x = self.encoder(x)
        age = torch.sigmoid(self.fc1(x))
        gender = self.fc2(x)
        mask = self.fc3(x)

        return [age, gender, mask]