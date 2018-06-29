'''
ALL_CNN main script, you can choose either 
- FashionMNIST 
- CIPAR10
images dataset, to try the model
set up 
importing model, model paprameters, and image_depth (channels)
'''

import os
import torch
from torch.autograd import Variable
import torchvision
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms
from torchvision.utils import save_image
from all_CNN_model import all_CNN
from logger import Logger


# Device configuration, cuda or cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.cuda.is_available() returns true if cuda is available, false otherwise


# Hyper Parameters
latent_size = 64 #what's this??
lr = 0.04 #0.25, 0.01, 0.05, , 
#hidden_size = 256 
image_size = 32# for whatever image's hight and witdh
num_epochs = 50
num_classes = 10
batch_size = 64 
image_depth = 3
sample_dir = 'CIFAR10_sample' # Forlder for data or log??

# Initialize logger
logPath = 'logs_CNN/'
record_name = 'CIFAR10_' + str(lr)
logger = Logger(logPath + record_name)

# Create a directory if not exist
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


# Image processing
# https://pytorch.org/docs/stable/torchvision/transforms.html
# transforms.Normalize(mean=(), std=()) normalize image with given means and 
# standard div, one value for each channel, here 3 for RBG
transform = transforms.Compose([ # transforms.Compose, list of transforms to perform
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
#
# MNIST dataset
#dataset = torchvision.datasets.FashionMNIST(root='data/', # where at??
#                                   train=True,
#                                   transform=transform, # pass the transform  we made
#                                   download=True)

train_dataset = torchvision.datasets.CIFAR10(root='CIFAR10_data/', # where at??
                                   train=True,
                                   transform=transform, # pass the transform  we made
                                   download=True)

test_dataset = torchvision.datasets.CIFAR10(root='CIFAR10_data/', # where at??
                                   train=False,
                                   transform=transform, # pass the transform  we made
                                   download=True)
# Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True, drop_last=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False, drop_last=True)
##### Models ######

D = all_CNN(image_depth, num_classes)
#G = Generator(image_depth, latent_size)
#
# Device setting
# D.to() moves and/or casts the parameters and buffers to device(cuda), dtype
# setting to whatevefr the device set earlier
D = D.to(device)
#G = G.to(device)

# BInary cross entropy loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(D.parameters(), lr=lr, momentum=0.9, weight_decay=0.001)
#g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

## Adaptive Learning Rate ##
scheduler = MultiStepLR(optimizer, milestones=[200, 250, 300], gamma=0.1)


#### some util functions? ####
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# out.clamp() filter the values within the specified range??
# out.clamp(min=0) is ReLU

## Zeros the gradient for both model's optimizers
#def reset_grad():
#    d_optimizer.zero_grad()
#    g_optimizer.zero_grad()
#

def evaluate(mode, num):
    '''
    Evaluate using only first num batches from loader
    '''
    #D.eval()
    test_loss = 0
    correct = 0
    if mode == 'train':
        loader = train_loader
    elif mode == 'test':
        loader = test_loader

    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
            data, target = Variable(data), Variable(target)
            output = D(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            if i % 10 == 0:
                print(i)
            if i == num: # break out when numth number of batch
                break
        sample_size = batch_size * num
        test_loss /= sample_size
        print('\n' + mode + 'set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, sample_size,
            100. * correct / sample_size))
    return 100. * correct / sample_size


#### Start Training ####
# num of batches in the total dataset, here 60000/100
total_step = len(train_loader) 

count = 0
for epoch in range(num_epochs): # How many times to go through the dataset
    D.train()
    for i, (images, labels) in enumerate(train_loader): # each batch
        count += 1
        # i: index num for the for loop
        # images: image matrix, size(batch x channels x 28 x 28)
        # _ : label, 0~9, size (batch,) 
        images = images.reshape(batch_size, image_depth, image_size, image_size).to(device) # reshape and set to cuda/cpu

        ### Create the labels which are later used as input for the BCE loss

        ## 1 for real, 0 for fake pic
        #real_labels = torch.ones(batch_size, 1).to(device) #(batch x 1)
        #fake_labels = torch.zeros(batch_size, 1).to(device)

        ######## Train the Discriminator #########

        # Compute BCELoss using real images 
        # Second term of the loss is always zero since real_labels == 1.. WHY??
        outputs = D(images) # using real data
        loss = criterion(outputs, labels)
        # Accuracy

        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0 ???
        #z = torch.randn(batch_size, latent_size).to(device) # create random noise
        #fake_images = G(z) # create fake image from Generator, using the noise z
        #outputs = D(fake_images) # making pred using fake image
        #d_loss_fake = criterion(outputs, fake_labels)
        #fake_score = outputs

        # Backprop and optimize
        #d_loss = d_loss_real + d_loss_fake # adding up both loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ##### Train the Generator ######
        
        # compute loss with fake images
        #z = torch.randn(batch_size, latent_size).to(device) # generate random noise
        #fake_images = G(z) # making fake images
        #outputs = D(fake_images) # output from Discriminator?? again?


        # We train G to maximize log(D(G(z))) instead of minimizing log(1-D)(G(z))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
        ### making prediction with fake images, and comparing the output to real
        # training so that fake_images, G(z) is as close as real_label
        #g_loss = criterion(outputs, real_labels)

        # Backprop and optimize, only Generator, based on discriminators loss
        if (i+1) % 2 == 0:
            print('Epoch [{}/{}], Step [{}/{}], loss: {:.4f}'.format(epoch, num_epochs, i+1, total_step, loss.item()))
            print('i+1', i+1, ' Lr:', lr)


        ##### Tensorboard Logging ##### 
        if i % 10 == 0:
            # 1. Log scalar values (scalar summary)
            info = {'loss':loss.item()}

            for tag, value in info.items():
                logger.scalar_summary(tag, value, count+1)


            # 2. Log values and gradients of the parameters (histogram summary)
            for tag, value in D.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), count+1)
                logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), i+1)

            print('logging on tensorboard...', i)

        if (i+1) % 100 == 0:
            print('calculating accuracy....')
            train_acc = evaluate('train', 50)
            test_acc = evaluate('test', 50)

            info = {'Train_acc':train_acc, 'Test_acc':test_acc}

            for tag, value in info.items():
                logger.scalar_summary(tag, value, count+1)

# Save the model checkpoints
torch.save(D.state_dict(), os.path.join(sample_dir, 'D.ckpt'))
    

