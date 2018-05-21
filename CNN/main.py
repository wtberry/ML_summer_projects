'''
Practice using pytorch for CNN, following tutorial on youtube,
link: https://www.youtube.com/watch?v=LgFNRIFxuUo&t=5s 
'''

### Importing libraries
from __future__ import print_function
from model import CNN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


##### Setting up parameters
batch_size = 100
lr = 0.01

##### Loading and preparing the fashionMNIST dataset

## Loading datasets
train_dataset = datasets.FashionMNIST(root='.data/', train=True,
                                        transform=transforms.ToTensor(), download=True)

test_dataset = datasets.FashionMNIST(root='.data/', train=False,
                                        transform=transforms.ToTensor())

## Creating dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size, shuffle=False)



