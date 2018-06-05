## Sentiment Analysis Practice

- **Dataset:** movie review from NLTK corpus
- Word2Vec were created using gensim
- LSTM model using Pytorch
	- number of layers: 1, 2, 3
	- memory/hidden layer size: 32, 
- Data Preprocessing:
	- few Convolutional layers before LSTM
	- feature normalization
	- **Principal Component Analysis: Works the best**
	- seqence length after PCA: 10, 15, 20, 25, 30
- Cost: CrossEntropyLoss
- optimizer: SGD, Adagrad, AdaDelta, ASGD, Adamax
	- SGD: good??
	- Adam: Nope
	- **Adagrad: works better**
	- AdaDelta: soso
	- RMPprop
	- ASGD: soso
	- Rprop: trash
	- Adamax: not bad, went almost as high as the best record around 1000 epoch, maybe adjust params?
- learning rate and scheduling
	* 0.03


Start with dif Layers
