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
epoch = 2
PATH = '/home/wataru/ML_summer_projects/CNN/save'

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


##### Define model, loss function, and optimizer
model = CNN()
optimizer = optim.SGD(model.parameters(), lr=lr)
criterion = nn.NLLLoss()

##### Training
def train(epoch):
    model.train() # set the model for training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.data[0]))

##### Testing
def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True) # set Volatile to true, when not running backprop
        output = model(data)
        test_loss += criterion(output, target).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(epoch): ## loop over the dataset # epoch times
    train(epoch)
    test()
#print('saving model...')
#torch.save(model.state_dict(), PATH)
