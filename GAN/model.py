import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):

    def __init__(self, image_size, hidden_size, drop):
        super(Discriminator, self).__init__()

        self.image_size = image_size
        self.hidden_size = hidden_size
        self.drop = drop

        self.linear1 = nn.Linear(self.image_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size, 1)

        
    def forward(self, x):
        x = F.leaky_relu(self.linear1(x), negative_slope=0.2)
        x = F.leaky_relu(self.linear2(x), negative_slope=0.2)
        out = F.sigmoid(self.linear3(x))
        return out



class Generator(nn.Module):

    def __init__(self, image_size, hidden_size, latent_size):
        super(Generator, self).__init__()

        self.image_size = image_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.linear1 = nn.Linear(self.latent_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size, self.image_size)

        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        out = F.tanh(self.linear3(x))
        return out

