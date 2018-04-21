import torch
import torchvision.models
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import sys
import os

# model collections
class fcn32s(nn.Module):
    def __init__(self, num_classes, pretrained = True):
        super(fcn32s, self).__init__()
        self.vgg = torchvision.models.vgg16(pretrained=True)
        # nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, 
        # output_padding=0, groups=1, bias=True, dilation=1)
        self.vgg.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=(2, 2), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            
            nn.Conv2d(4096, 4096, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            
            nn.Conv2d(4096, num_classes, kernel_size=(1, 1), stride=(1, 1)),
            nn.ConvTranspose2d(num_classes, num_classes, 64 , 32 , 0, bias=False),
        )
    def  forward (self, x) :        
        x = self.vgg.features(x)
#         print(x.size())
        x = self.vgg.classifier(x)
        return x

class fcn32s_prune(nn.Module):
    # cut off the conv6 and conv7, then directly upsample
    def __init__(self, num_classes, pretrained = True):
        super(fcn32s_prune, self).__init__()
        self.vgg = torchvision.models.vgg16(pretrained=True)
        self.vgg.classifier = nn.Sequential(
            nn.ConvTranspose2d(512, num_classes, 32 , 32 , 0, bias=False)
        )
    def  forward (self, x) :        
        x = self.vgg.features(x)
        x = self.vgg.classifier(x)
        return x

class fcn16s(nn.Module):
    def __init__(self, num_classes, pretrained = True):
        super(fcn16s, self).__init__()
        self.vgg = torchvision.models.vgg16(pretrained=True)
        self.to_pool4 = nn.Sequential(*list(self.vgg.features.children())[:24])
        self.to_pool5 = nn.Sequential(*list(self.vgg.features.children())[24:])
        self.vgg.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=(2, 2), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            
            nn.Conv2d(4096, 4096, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            
            nn.Conv2d(4096, num_classes, kernel_size=(1, 1), stride=(1, 1)),
            nn.ConvTranspose2d(num_classes, 512, 4 , 2 , 0, bias=False)
            )
        self.upsample16 = nn.ConvTranspose2d(512, num_classes, 16 , 16 , 0, bias=False)
    def  forward (self, x) :        
        pool4_output = self.to_pool4(x) #pool4 output size torch.Size([64, 512, 16, 16])
        x = self.to_pool5(pool4_output)
        x = self.vgg.classifier(x)    # 2xconv7 output size torch.Size([64, 512, 16, 16])
        x = self.upsample16(x+pool4_output)
        return x

class fcn8s(nn.Module):
    def __init__(self, num_classes, pretrained = True):
        super(fcn8s, self).__init__()
        self.vgg = torchvision.models.vgg16(pretrained=True)
        self.to_pool3 = nn.Sequential(*list(self.vgg.features.children())[:17])
        self.to_pool4 = nn.Sequential(*list(self.vgg.features.children())[17:24])
        self.to_pool5 = nn.Sequential(*list(self.vgg.features.children())[24:])
        self.vgg.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=(2, 2), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            
            nn.Conv2d(4096, 4096, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            
            nn.Conv2d(4096, num_classes, kernel_size=(1, 1), stride=(1, 1)),
            nn.ConvTranspose2d(num_classes, 256, 8 , 4 , 0, bias=False) # 4x conv7
            )
        self.pool4_upsample = nn.ConvTranspose2d(512, 256, 2 , 2 , 0, bias=False)
        self.upsample8 = nn.ConvTranspose2d(256, num_classes, 8 , 8 , 0, bias=False)
    def  forward (self, x) :
        pool3_output = self.to_pool3(x) # [64, 256, 32, 32]
        pool4_output = self.to_pool4(pool3_output) #pool4 output size torch.Size([64, 512, 16, 16])
        pool4_2x = self.pool4_upsample(pool4_output) # 2x pool4 torch.Size([64, 512, 32, 32])
        x = self.to_pool5(pool4_output)
        x = self.vgg.classifier(x)  # 4x conv7 torch.Size([64, 256, 32, 32])
        x = self.upsample8(x+pool3_output+pool4_2x)
        return x