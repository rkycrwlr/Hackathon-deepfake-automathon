import torch
import torch.nn as nn
import torchvision.models as models


class EffNetB0(nn.Module):
    def __init__(self, out_size=1):
        """Load the pretrained EfficientNet-B0 and replace top fc layer."""
        super(EffNetB0, self).__init__()
        cnn = models.efficientnet_b0(weights='IMAGENET1K_V1')
        modules = list(cnn.children())[:-1]     
        self.cnn = nn.Sequential(*modules)
        self.linear = nn.Linear(1280, out_size)
        self.sig = nn.Sigmoid()
        self.name = "EffNetB0"
        # self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        features = self.cnn(images)
        features = features.reshape(features.size(0), -1)
        # out = self.sig(self.linear(features))
        return self.linear(features)
    
class Resnet50(nn.Module):
    def __init__(self, out_size=1):
        """Load the pretrained Resnet50 and replace top fc layer."""
        super(Resnet50, self).__init__()
        cnn = models.resnet50(weights='IMAGENET1K_V2')
        modules = list(cnn.children())[:-1]
        self.cnn = nn.Sequential(*modules)
        self.linear = nn.Linear(2048, out_size)
        self.sig = nn.Sigmoid()
        self.name = "Resnet50"
        # self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        features = self.cnn(images)
        features = features.reshape(features.size(0), -1)
        #out = self.sig(self.linear(features))
        return self.linear(features)
