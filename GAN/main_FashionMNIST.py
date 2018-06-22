import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image


# Device configuration, cuda or cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.cuda.is_available() returns true if cuda is available, false otherwise

# Hyper Parameters
latent_size = 64 #what's this??
hidden_size = 256 # for??
image_size = 784 # MNIST??
num_epochs = 200
batch_size = 100
sample_dir = 'fashionMNIST_samples' # Forlder for data or log??


# Create a directory if not exist
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


# Image processing
# https://pytorch.org/docs/stable/torchvision/transforms.html
# transforms.Normalize(mean=(), std=()) normalize image with given means and 
# standard div, one value for each channel, here 3 for RBG
transform = transforms.Compose([ # transforms.Compose, list of transforms to perform
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), # 3 for RGB channels
                                     std=(0.5, 0.5, 0.5))])

# MNIST dataset
fmnist = torchvision.datasets.FashionMNIST(root='data/', # where at??
                                   train=True,
                                   transform=transform, # pass the transform  we made
                                   download=True)

# Data Loader
data_loader = torch.utils.data.DataLoader(dataset=fmnist,
                                          batch_size=batch_size,
                                          shuffle=True)


##### Models ######

# Discriminator
# Fully connected layers with LeakyReLU, and sigmoid on top
## Activation function choice based on paper??
# paper: DCGAN https://arxiv.org/abs/1511.06434
D = nn.Sequential(
        nn.Linear(image_size, hidden_size),
        nn.LeakyReLU(0.2),
        nn.Linear(hidden_size, hidden_size),
        nn.LeakyReLU(0.2),
        nn.Linear(hidden_size, 1),
        nn.Sigmoid())

# Generator
G = nn.Sequential(
        nn.Linear(latent_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, image_size),
        nn.Tanh()) # why tahn??


# Device setting
# D.to() moves and/or casts the parameters and buffers to device(cuda), dtype
# setting to whatevefr the device set earlier
D = D.to(device)
G = G.to(device)

# BInary cross entropy loss and optimizer
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)


#### some util functions? ####
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# out.clamp() filter the values within the specified range??
# out.clamp(min=0) is ReLU

## Zeros the gradient for both model's optimizers
def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()


#### Start Training ####
# num of batches in the total dataset, here 60000/100
total_step = len(data_loader) 

for epoch in range(num_epochs): # How many times to go through the dataset
    for i, (images, _) in enumerate(data_loader): # each batch
        # i: index num for the for loop
        # images: image matrix, size(batch x channels x 28 x 28)
        # _ : label, 0~9, size (batch,) 
        images = images.reshape(batch_size, -1).to(device) # reshape and set to cuda/cpu

        ### Create the labels which are later used as input for the BCE loss

        ## 1 for real, 0 for fake pic
        real_labels = torch.ones(batch_size, 1).to(device) #(batch x 1)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        ######## Train the Discriminator #########

        # Compute BCELoss using real images 
        # Second term of the loss is always zero since real_labels == 1.. WHY??
        outputs = D(images) # using real data
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0 ???
        z = torch.randn(batch_size, latent_size).to(device) # create random noise
        fake_images = G(z) # create fake image from Generator, using the noise z
        outputs = D(fake_images) # making pred using fake image
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        # Backprop and optimize
        d_loss = d_loss_real + d_loss_fake # adding up both loss
        reset_grad() # grad.zeros()
        d_loss.backward()
        d_optimizer.step()

        ##### Train the Generator ######
        
        # compute loss with fake images
        z = torch.randn(batch_size, latent_size).to(device) # generate random noise
        fake_images = G(z) # making fake images
        outputs = D(fake_images) # output from Discriminator?? again?


        # We train G to maximize log(D(G(z))) instead of minimizing log(1-D)(G(z))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
        ### making prediction with fake images, and comparing the output to real
        # training so that fake_images, G(z) is as close as real_label
        g_loss = criterion(outputs, real_labels)

        # Backprop and optimize, only Generator, based on discriminators loss
        reset_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i+1) % 200 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'.format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(),
                real_score.mean().item(), fake_score.mean().item()))


    # Save real images
    if (epoch+1) == 1:
        images = images.reshape(images.size(0), 1, 28, 28)
        save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'))

    
    # Save sampled images
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))

# Save the model checkpoints
torch.save(G.state_dict(), 'G.ckpt')
torch.save(D.state_dict(), 'D.ckpt')
    

