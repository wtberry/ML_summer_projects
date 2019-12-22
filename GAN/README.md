### Tutorial and experiments with GAN & CNN

GAN and CNN were experimented in this directory, for the All Convolutional Net and DCGAN(Deep Convolutional Generative Adversarial Networks)

- tutorial on github, about GAN using pytorch
 https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/generative_adversarial_network/main.py#L41-L57

### Researches that I inpremented
- DCGAN: deep convolutional generative
adversarial networks
- STRIVING FOR SIMPLICITY: THE ALL CONVOLUTIONAL NET
- Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

### main files
- main_CIFAR10.py: DCGAN using CIFAR10 training script
- main_FashionMNIST.py: plain GAN using fashionMNIST dataset
- main_CNN.py: all_CNN training using CIFAR10
- main_MNIST.py: plain GAN, using MNIST

### Models
- all_CNN_model.py: all_CNN model, fancy CNN based on the paper
- DCGAN_model.py: DCGAN model, using all_CNN above
- model.py: plain fully connected GAN
