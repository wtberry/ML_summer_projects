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
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5) #(input_depth, # of fileters)
        self.conv2 = nn.Conv2d(20, 20, kernel_size=5)
        self.mp = nn.MaxPool2d(2) # maxpool layer
        self.dense = nn.Linear(320, 10) # number of class is 10


    # forward pass
    def forward(self, x):
        input_size = x.size(0)
        c1 = self.conv1(x)
        mp1 = self.mp(c1)
        relu1 = F.relu(mp1)

        c2 = self.conv2(relu1)
        mp2 = self.mp(c2)
        relu2 = F.relu(mp2)
        out = relu2.view(in_size, -1) # flatten the output for Dense layer

        out = self.dense(out)
        return F.log_softmax(out)

