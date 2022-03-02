import torch
import torch.nn as nn
import timm

# ConvNext Base : 224x224
class convnext(nn.Module):
    def __init__(self, num_classes, freeze = True, drop_rate = 0.7):
        super(convnext, self).__init__()

        self.convnext = timm.create_model('convnext_base', pretrained=True, num_classes = num_classes, drop_rate=0.5)

        if freeze == True:
            timm.utils.freeze(self.convnext)
            self.convnext.fc.weight.requires_grad = True
            self.convnext.fc.bias.requires_grad = True

    def forward(self, x):
        return self.convnext(x)

class coatnet(nn.Module):
    def __init__(self, num_classes, freeze = True, drop_rate = 0.7):
        super(coatnet, self).__init__()

        self.coatnet = timm.create_model('coat_mini', pretrained=True, num_classes = num_classes, drop_rate=0.5)

        if freeze == True:
            timm.utils.freeze(self.coatnet)
            self.coatnet.fc.weight.requires_grad = True
            self.coatnet.fc.bias.requires_grad = True

    def forward(self, x):
        return self.coatnet(x)


# EfficientNet B6 : 528x528
# b0 : 224x224
class effnet(nn.Module):
    def __init__(self, num_classes, freeze = True, drop_rate = 0.7):
        super(effnet, self).__init__()

        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)

        if freeze == True:
             timm.utils.freeze(self.efficientnet)
             self.efficientnet.classifier.weight.requires_grad = True
             self.efficientnet.classifier.bias.requires_grad = True

    def forward(self, x):
        return self.efficientnet(x)


# ResNext101 32x8d : 224x224
class resnext(nn.Module):
    def __init__(self, num_classes, freeze = True, drop_rate=0.7):
        super(resnext, self).__init__()

        self.resnext = timm.create_model('resnext101_32x8d', pretrained=True, num_classes=num_classes)
        
        if freeze == True:
            timm.utils.freeze(self.resnext)
            self.resnext.fc = nn.Sequential(
                            nn.Linear(2048, 512),
                            nn.ReLU(),
                            nn.Linear(512, num_classes),
                            nn.Sigmoid())


    def forward(self, x):
        return self.resnext(x)

