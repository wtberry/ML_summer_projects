from model import LSTM
from logger import Logger
from create_seq import reviews

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt


'''
LSTM for natural language processing, sentiment analysis using
- movie review dataset from NLTK
- Word2Vec from gensim
- LSTM and optimization with pytorch
'''

torch.manual_seed(777)

##### Parameters and loading data #####
seq_len = 1000
data = reviews(seq_len)
num_classes = 2
input_size = data.input_size
hidden_size = 32 
num_layers = 1



valid_percentage = 0
train_percentage = 90
pca_comp = 20

batch_size = 100

lr = 0.03
num_iter = int(1e+4)


##### Data Loading and Prep #####
X = data.X
y = data.y

## normalize X
#X = data.normalize(X)

## Applying PCA
X = data.pca(X, pca_comp)
print('X after PCA: ', X.shape)

# Split the data into train, valid, and test set
data_dict = data.data_split(train_perc=train_percentage, valid_perc=valid_percentage, X=X, y=y)
X_train, X_valid, X_test = data_dict['train_X'], data_dict['valid_X'], data_dict['test_X']
y_train, y_valid, y_test = data_dict['train_label'], data_dict['valid_label'], data_dict['test_label']


# We'll convert the arrays into torch tensors
X_train_tensor, X_valid_tensor, X_test_tensor  = torch.Tensor(X_train), torch.Tensor(X_valid), torch.Tensor(X_test)
y_train_tensor, y_valid_tensor, y_test_tensor = torch.Tensor(y_train), torch.Tensor(y_valid), torch.Tensor(y_test)

# Then put the tensors into dataset
train_set = TensorDataset(X_train_tensor, y_train_tensor)
valid_set = TensorDataset(X_valid_tensor, y_valid_tensor)
test_set = TensorDataset(X_test_tensor, y_test_tensor)

# Finally create data loader
train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=True)
#valid_loader = DataLoader(valid_set, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test_set, batch_size=batch_size, drop_last=True)



##### Set up the Logger #####


##### Instantiate the LSTM and Prep for Training #####
# model
lstm = LSTM(num_classes, input_size, hidden_size, num_layers)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(lstm.parameters(), lr=lr)


##### Defining Utility Functions #####

def set_log():
    log_path = '/home/wataru/ML_summer_projects/nltk/sentiment/log/'
    log_dir = log_dir= 'PCA' + str(pca_comp) +  '_layer_'+ str(num_layers)+ '_memory_' + str(hidden_size) + '_lr_'+str(lr)+'_epoch_'+str(num_iter) + '_optim_' + str(optimizer).split(' ')[0]
    logger = Logger(log_path + log_dir)
    return logger, log_path, log_dir

def to_np(x):
    return x.data.cpu().numpy()

def accuracy(prediction, labels):
    _, idx = prediction.max(1)
    idx_label = labels.vew(seq_len*batch_size)
    compare = idx == idx_label
    compare = compare.float()
    return compare.mean() * 100

def eval_plot(data, target):
    output = lstm(data)
    fig, ax = plt.subplots()
    ax.plot(to_np(target), label='label')
    ax.plot(to_np(output), label='prediction')
    fig.legend()
    fig.show()

def test():
    lstm.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target).type(torch.LongTensor)
        output = lstm(data)
        test_loss += criterion(output, target).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    
    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))
    return test_loss, acc

def print_params():

    # Print model info
    print('*'*10, 'model info', '*'*10)
    print('Num of layers: ', num_layers)
    print('seq_length: ', seq_len)
    print('# of features:', input_size)
    print('batch_size: ', batch_size)
    print(lstm)

    # Print training info
    print('*'*10, 'training info', '*'*10)
    print('Learning rate: ', lr)
    print('optimizer: ', str(optimizer).split(' ')[0])


##### Training #####
# making function calls
logger, _, _ = set_log()
print_params()
lstm.train()

epoch = 0
while epoch < num_iter:
    for (data, target) in train_loader:
        if epoch == num_iter:
            break
        else:
            data, target = Variable(data), Variable(target).type(torch.LongTensor)
            optimizer.zero_grad()
            outputs = lstm(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
	    
            epoch +=1

            if (epoch+1)%100==0:

                # printing out accuracyy
                prediction = outputs.max(dim=1, keepdim=True)[1]
                print()
                print("epoch: %d, loss: %1.3f" % (epoch + 1, loss.item()))
                #print("training_accuracy: %1.3f" % (acc_train))
                print("predicted value: ", prediction[0].item())
                print("label value:     ", target[0].item())


                ##### tensorboard logging
                # 1. log the scalar values
                info = {
                        'loss': loss.item(),
                        #'accuracy': acc_train
                        }
                
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, epoch+1)
                
                
                # 2. log values and gradients of the parameters (histogram)
                for tag, value in lstm.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, to_np(value), epoch+1)
                    logger.histo_summary(tag+'/grad', to_np(value.grad), epoch+1)

            ################ call Test?? #####
            if (epoch+1)%500==0:
                '''
                # make test prediction
                test_out = lstm(X_test)
                test_loss = criterion(test_out, y_test)
                print()
                print("epoch: %d, test loss: %1.3f" % (epoch+1, test_loss.item()))
                print("predicted value: ", test_out.data[0])
                print("label value:     ", y_test.data[0])
                '''
                test_loss, accuracy = test()
                info = {
                        'test loss': test_loss.item(),
                        'test accuracy': accuracy.item()
                        }

                for tag, value in info.items():
                    logger.scalar_summary(tag, value, epoch+1)

print("learning finished!")

##### Testing Accuracy #####
#prediction = lstm()
