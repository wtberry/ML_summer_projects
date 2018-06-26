import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Exprimenting GAN with CNN, some config are based on DCGAN Paper
- conv layer instead of MaxPool
- No FC layers, global average pooling layer instead
~~ Global Average pooling https://www.quora.com/What-is-global-average-pooling
- batch normalization, input to have zero mean and unit variance
- Activations: ReLU for Generator, Leaky ReLU for discriminator


# Example of well-modulized beautiful CNN
https://heartbeat.fritz.ai/basics-of-image-classification-with-pytorch-2f8973c51864
'''

class CUnit(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, batch_norm=True):
        super(CUnit, self).__init__()
        pass

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, inp, batch_norm=True):
        out = self.conv(inp)
        if batch_norm:
            out = self.bn(out)
        out = self.lrelu(out)

        return out



class Discriminator(nn.Module):

    def __init__(self, image_depth, num_classes):
        super(Discriminator, self).__init__()

        self.image_depth = image_depth
        self.num_classes = num_classes

        self.num_out1 = 32
        self.num_out2 = 64

        ## Create units
        self.unit1 = CUnit(in_channels=self.image_depth, out_channels=64, stride=1, batch_norm=False) 
        # (16x16)
        self.unit2 = CUnit(in_channels=64, out_channels=64, kernel_size=5, stride=1)
        # (8x8)
        self.unit3 = CUnit(in_channels=64, out_channels=64, kernel_size=5, stride=2)
        # (4x4)
        self.unit4 = CUnit(in_channels=64, out_channels=32, kernel_size=1, stride=1)
        self.unit5 = CUnit(in_channels=32, out_channels=32, kernel_size=1, stride=1)
        self.unit6 = CUnit(in_channels=32, out_channels=self.num_classes, kernel_size=3, stride=2)

        # batch_norm https://discuss.pytorch.org/t/example-on-how-to-use-batch-norm/216/2
       
        ### Change nn.AvgPool2d(x), x to 3 if MNIST, 4 for CIFAR10
        self.avp= nn.AvgPool2d(4) 
        self.softmax = nn.Softmax(dim=1)
        #self.conv1 = nn.Conv2d()

    def forward(self, x):
        # batch norm
        #print('norm', x[0, :, :, :])
        # three conv layers
        #print('x', x.shape)
        x = self.unit1(x)
        #print('1', x.shape)
        x = self.unit2(x)
        #print('2', x.shape)
        x = self.unit3(x)
        #print('3th output: ', x.shape) # (batch x 10 x 9 x 9)
        x = self.unit4(x)
        #print('4th output: ', x.shape) # (batch x 10 x 9 x 9)
        x = self.unit5(x)
        #print('5th output: ', x.shape) # (batch x 10 x 9 x 9)
        x = self.unit6(x)
        #print('6th output: ', x.shape) # (batch x 10 x 9 x 9)
        # global average pooling
        #avg1 = x.mean(dim=2)
        #avg2 = x.mean(dim=2) # (batch x classes)
        avg = self.avp(x) # (batch x channel x 1 x 1)
        #print('avg', avg.shape)
        avg = avg.view(-1, self.num_classes) # (batch x classes)
        # softmax and reshape into label shape
        # has to be max()[0], for it to have grad for backprop
        out = self.softmax(avg).max(dim=1, keepdim=True)[0]
        #print('out', out.shape)
        return out


class FUnit(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, padding=2, out_padding=1, out_layer=False):
        super(FUnit, self).__init__()
        pass

        self.fconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=out_padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, inp, out_layer=False):
        out = self.fconv(inp)
        if out_layer==False:
            out = self.bn(out)
            out = self.relu(out)
        else:
            out = self.tanh(out)

        return out


'''
fractionally convolution
https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d
'''

class Generator(nn.Module):


    def __init__(self, image_depth, latent_size):
        super(Generator, self).__init__()

        self.latent_size = latent_size
        self.dense = nn.Linear(self.latent_size, 512*4*4)
        self.unit1 = FUnit(in_channels=512, out_channels=256, padding=2, out_padding=1) # (8x8)
        self.unit2 = FUnit(in_channels=256, out_channels=128) # (16x16)
        self.unit3 = FUnit(in_channels=128, out_channels=image_depth, out_layer=True) #(32x32)

    def forward(self, x):
        # some projection and reshaping of uniform distro, using dense layer
        proj  = self.dense(x) 
        x = proj.reshape(-1, 512, 4, 4)
        #print('x', x.shape)
        x = self.unit1(x)
        #print('x1', x.shape)
        x = self.unit2(x)
        #print('x2', x.shape)
        out = self.unit3(x)
        #print('Gen_out', out.shape)

        return out
