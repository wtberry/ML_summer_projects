'''
CNN model using pytorch
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        # Defining layers needed
        self.conv1 = nn.Conv2d(1, 70, kernel_size=5) #(input_depth, # of fileters)
        self.conv2 = nn.Conv2d(70, 20, kernel_size=3)
        self.mp = nn.MaxPool2d(2) # maxpool layer
        self.dense = nn.Linear(500, 10) # number of class is 10


    # forward pass
    def forward(self, x):
        in_size = x.size(0)
        c1 = self.conv1(x)
        mp1 = self.mp(c1)
        relu1 = F.leaky_relu(mp1)

        c2 = self.conv2(relu1)
        mp2 = self.mp(c2)
        relu2 = F.leaky_relu(mp2)
        out = relu2.view(in_size, -1) # flatten the output for Dense layer

        out = self.dense(out)
        return out 

