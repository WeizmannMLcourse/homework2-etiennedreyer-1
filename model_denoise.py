
import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict

class DenoiseNet(nn.Module):

    def __init__(self):
        super().__init__()
        
        ### Add your network layers here...
        ### You should use nn.Conv2d(), nn.BatchNorm2d(), and nn.ReLU()
        ### They can be added as seperate layers and cascaded in the forward
        ### or you can combine them using the nn.Sequential() class and an OrderedDict (very clean!)
        
    def forward(self,x):
        
        ### Now pass the input image x through the network layers
        ### Then add the result to the input image (to offset the noise)

        return ### the sum ###