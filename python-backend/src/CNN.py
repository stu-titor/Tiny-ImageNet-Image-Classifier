import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import SqueezeExcitation
from torchvision.ops import stochastic_depth

class ImageNeuralNetwork(nn.Module):
    
    def __init__(self, channels = 32, layers = 3, conv_blocks = 4, num_classes = 10, se_reduction=16, dropout_rate = 0.35, max_sd_prob = 0.15):
        super().__init__() 
        
        # Instance variables
        self.convLayers1 = nn.ModuleList()
        self.batches1 = nn.ModuleList()

        self.convLayers2 = nn.ModuleList()
        self.batches2 = nn.ModuleList()

        self.seBlocks = nn.ModuleList() 
        self.fcs = nn.ModuleList()

        # Convolutional layers and batch normalizations
        input_channels = 3
        output_channels = channels
        
        for i in range(conv_blocks):
            # first conv batch pair
            self.convLayers1.append(nn.Conv2d(input_channels, output_channels, 3, padding = 1))
            self.batches1.append(nn.BatchNorm2d(output_channels))

            # second conv batch pair
            self.convLayers2.append(nn.Conv2d(output_channels, output_channels, 3, padding = 1))
            self.batches2.append(nn.BatchNorm2d(output_channels))

            squeeze_channels = max(output_channels // se_reduction, 1)
            self.seBlocks.append(SqueezeExcitation(output_channels, squeeze_channels))
            
            input_channels = output_channels
            output_channels *= 2

        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        input_size = output_size = input_channels 

        for i in range(layers - 1):
            self.fcs.append(nn.Linear(input_size, output_size)) 
            input_size = output_size
            output_size //= 2
        self.fcs.append(nn.Linear(input_size, num_classes))

        self.dropout = nn.Dropout(dropout_rate)

        self.sd_probability = []
        for i in range(conv_blocks):
            self.sd_probability.append(max_sd_prob * i / max(conv_blocks - 1, 1))

    def forward(self, x):
        # Convolutional blocks
        for i in range(len(self.convLayers1)):
            inital = x
            x = F.silu(self.batches1[i](self.convLayers1[i](x)))
            x = self.batches2[i](self.convLayers2[i](x))
            x = self.seBlocks[i](x) 

            if i != 0:
                if inital.shape[1] != x.shape[1]:
                    inital = F.pad(inital, (0,0,0,0,0, x.shape[1] - inital.shape[1]))
                x = stochastic_depth(x, p = self.sd_probability[i], mode = "batch", training = self.training) 
                x += inital
            x = F.silu(x)
            x = self.pool(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        
        # Fully connected layers with dropout
        for i in range(len(self.fcs) - 1):
            x = self.dropout(F.silu(self.fcs[i](x)))
        x = self.fcs[-1](x) 
        
        return x