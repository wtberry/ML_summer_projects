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
- optimizer:
	- SGD
	- **Adagrad: works better**
- learning rate and scheduling
	* 0.03
