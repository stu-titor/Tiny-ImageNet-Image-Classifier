import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageNeuralNetwork(nn.Module):
    
    def __init__(self, channels = 32, layers = 3, conv_blocks = 4, num_classes = 10, image_size=32, dropout_rate = 0.5):
        super().__init__() 
        
        # Instance variables
        self.convBlocks = nn.ModuleList()
        self.batches = nn.ModuleList()
        self.fcs = nn.ModuleList()

        #Convolutional blocks and batch normalizations
        input_channels = 3
        output_channels = channels
        
        for i in range(conv_blocks):
            self.convBlocks.append(nn.Conv2d(input_channels, output_channels, 3, padding = 1))
            self.batches.append(nn.BatchNorm2d(output_channels))
            input_channels = output_channels
            output_channels *= 2
    
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        final_size = image_size // (2 ** conv_blocks)
        input_size = input_channels * final_size * final_size
        output_size = input_channels

        for i in range(layers - 1):
            self.fcs.append(nn.Linear(input_size, output_size)) 
            input_size = output_size
            output_size //= 2
        self.fcs.append(nn.Linear(input_size, num_classes))

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Convolutional blocks
        for i in range(len(self.convBlocks)):
            x = self.pool(F.relu(self.batches[i](self.convBlocks[i](x))))
        x = torch.flatten(x, 1)
        
        # Fully connected layers with dropout
        for i in range(len(self.fcs) - 1):
            x = self.dropout(F.relu(self.fcs[i](x)))
        x = self.fcs[-1](x) 
        
        return x