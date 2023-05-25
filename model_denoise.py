
import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict

class DenoiseNet(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.features = nn.Sequential(OrderedDict([
            ('conv1',   nn.Conv2d(1,20,3,padding=1)),
            ('norm1',   nn.BatchNorm2d(20)),
            ('relu1',   nn.ReLU()),
            ('conv2',   nn.Conv2d(20,40,3,padding=1)),
            ('norm2',   nn.BatchNorm2d(40)),
            ('relu2',   nn.ReLU()),
            ('conv3',   nn.Conv2d(40,20,3,padding=1)),
            ('norm3',   nn.BatchNorm2d(20)),
            ('relu3',   nn.ReLU()),
            ('conv4',   nn.Conv2d(20,1,1,padding=0))
        ]))

        
    def forward(self,x):
        
        out  = self.features(x)
        out  = out + x
        return out