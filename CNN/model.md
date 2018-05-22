### Model Specs

#### Model Structures
- Conv layers: 2:
	- conv1(1, 70, kernel=5)
	- conv2(70, 20, kernel=3)
- MaxPool layers: 2:
	- mp(2)
- Activation: leaky_ReLU
- Dense layers: 1:
	- dense(500, 10)

#### Training tools/parameters
- Loss: CrossEntropyLoss

#### Results
- 4 loops on Tranining set
- Testing acc: 87%
