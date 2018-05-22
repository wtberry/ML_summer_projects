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
        fil_num = 30
        # Defining layers needed
        self.conv1 = nn.Conv2d(1, fil_num1, kernel_size=5) #(input_depth, # of fileters)
        self.conv2 = nn.Conv2d(fil_num1, 40, kernel_size=4)
        self.conv3 = nn.Conv2d(40, 40, kernel_size=3)
        self.conv4 = nn.Conv2d(40, fil_num, kernel_size=3)
        self.mp1 = nn.MaxPool2d(3, stride=1) # maxpool layer
        self.mp2 = nn.MaxPool2d(2) # maxpool layer
        self.mp3 = nn.MaxPool2d(2, stride=1) # maxpool layer
        self.dense1 = nn.Linear(875, 200) # number of class is 10
        self.dense2 = nn.Linear(480, 10) # number of class is 10


    # forward pass
    def forward(self, x):
        in_size = x.size(0)

        # 3 convolutions and maxpools
        c1 = self.conv1(x)
        #print('c1 size: ', c1.shape)
        mp1 = self.mp1(c1) # 100x20x12x12
        #print('size after first mp: ', mp1.shape)
        relu1 = F.leaky_relu(mp1)
        #print('relu1 size: ', relu1.shape)

        c2 = self.conv2(relu1)
        #print('c2 size: ', c2.shape)
        mp2 = self.mp1(c2) # 100x20x4x4
        #print('size after 2nd MP: ', mp2.shape)
        relu2 = F.leaky_relu(mp2)
        #print('relu2 size: ', relu2.shape)

        c3 = self.conv3(relu2)
        #print('c3 size: ', c3.shape)
        mp3 = self.mp2(c3)
        relu3 = F.leaky_relu(mp3)
        #print('relu3 size: ', relu3.shape)

        c4 = self.conv4(relu3)
        #print('c4 size: ', c4.shape)
        mp4 = self.mp3(c4)
        relu4 = F.leaky_relu(mp4)
        #print('relu4 size: ', relu4.shape)
        

        out = relu4.view(in_size, -1) # flatten the output for Dense layer
        #print('shape of out: ', out.shape)

        ## 2 dense layers with leaky ReLU
        #out = self.dense1(out)
        #out = F.leaky_relu(out)
        out = self.dense2(out)
        out = F.leaky_relu(out)
        return out

