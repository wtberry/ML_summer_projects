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

        # defining the params for the model
        fil_num1 = 50 
        fil_num = 20
        # Defining layers needed
        self.conv1 = nn.Conv2d(1, fil_num1, kernel_size=3) #(input_depth, # of fileters)
        self.conv2 = nn.Conv2d(fil_num1, 35, kernel_size=3)
        self.conv3 = nn.Conv2d(35, fil_num, kernel_size=3)
        self.mp = nn.MaxPool2d(2) # maxpool layer
        self.dense1 = nn.Linear(875, 200) # number of class is 10
        self.dense2 = nn.Linear(200, 10) # number of class is 10


    # forward pass
    def forward(self, x):
        in_size = x.size(0)

        # 3 convolutions and maxpools
        c1 = self.conv1(x)
        mp1 = self.mp(c1) # 100x20x12x12
        #print('size after first mp: ', mp1.shape)
        relu1 = F.leaky_relu(mp1)

        c2 = self.conv2(relu1)
        mp2 = self.mp(c2) # 100x20x4x4
        #print('size after 2nd MP: ', mp2.shape)
        relu2 = F.leaky_relu(mp2)

        c3 = self.conv3(relu2)
        mp3 = self.mp(c3)
        relu3 = F.leaky_relu(mp3)

        out = relu2.view(in_size, -1) # flatten the output for Dense layer
        #print('shape of out: ', out.shape)

        ## 2 dense layers with leaky ReLU
        out = self.dense1(out)
        out = F.leaky_relu(out)
        out = self.dense2(out)
        out = F.leaky_relu(out)
        return out

