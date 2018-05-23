### Model Specs

#### Model Structures
- Conv layers: 2/3
	* conv1: size 1, 64 
	* conv2: size 64, 64 
	* conv3: size 64, 32 
- MaxPool layers: 2
	* MaxPool: 2
- Activation: leaky_ReLU
- Dense layers: 2:
	* Dense(1600, 512)
	* Dense(512, 128)
	* Dense(128, 10)

#### Training tools/parameters
- Loss: CrossEntropyLoss
- epoch: 7 loops
- optimizer: Adagrad

#### Result
- Test set acc: 90%, avg loss: 0.0027, 7 epochs
