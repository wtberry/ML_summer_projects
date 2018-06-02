'''
LSTM model for NLP, sentiment analysis practice
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_layers = num_layers
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        self.conv1 = nn.Conv1d(input_size, input_size, kernel_size=6, stride=2)
        self.mp = nn.MaxPool1d(2)
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # Linear/ normal Neural Network layer on top of LSTM
        self.linear = nn.Linear(hidden_size, num_classes)
        # activation


    def forward(self, x):
        # reshape data for convolution
        batch_size = x.shape[0]
        x = x.view(batch_size, self.input_size, -1)
        
        ### Convolutional Layers ###
        c1 = self.conv1(x)
        mp1 = F.leaky_relu(self.mp(c1))
        c2 = self.conv1(mp1)
        mp2 = F.leaky_relu(self.mp(c2))

        # reshape again for LSTM
        mp2 = mp2.view(batch_size, -1, self.input_size)
        #print('mp2: ', mp2.shape)

        ### LSTM Layers ###
        # (num_layers * num_directions, batch hidden_size) for batch_first=True
        h_1 = (Variable(torch.randn(self.num_layers, x.size(0), self.hidden_size)), Variable(torch.randn(self.num_layers, x.size(0), self.hidden_size)))

        # Propagate input through RNN
        # Input: (batch, seq_len, input_size)
        # h_0: (batch, num_layers * num_directions, hidden_size)
        out, _ = self.lstm1(mp2, h_1)
        # adding up the output sequences
        out_sum = out.sum(dim=1)
        #print(out.shape)
        out = out_sum.contiguous().view(-1, self.hidden_size)

        # Neural Net/Fully Connected layer
        p = self.linear(out)
        p = F.sigmoid(p)
        return p

