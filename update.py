import torch
import torch.nn as nn
import math

class CorrelationDecoder(nn.Module):
    def __init__(self, args, input_dim=32, hidden_dim=64, output_dim=2, downsample=6):
        super(CorrelationDecoder, self).__init__()
        
        self.args = args
        self.in_conv = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)

        self.layers = nn.ModuleList([nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, stride=1), 
                                                   nn.GroupNorm(num_groups=(hidden_dim) // 8, num_channels=hidden_dim),
                                                   nn.ReLU(),
                                                   nn.MaxPool2d(kernel_size = 2, stride=2)) for i in range(downsample)])
        self.out_conv = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)
        
    def forward(self, corr, mask=0):
        x = self.in_conv(corr)
        for layer in self.layers: x = layer(x)
        x = self.out_conv(x)
        return x
        