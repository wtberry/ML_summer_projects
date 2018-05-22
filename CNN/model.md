### Model Specs

#### Model Structures
- Conv layers: 2/3
	* conv1: size 1, 20
	* conv2: size 20, 20
- MaxPool layers: 2
	* MaxPool: 2
- Activation: ReLU
- Dense layers: 1

#### Training tools/parameters
- Loss: CrossEntropyLoss
- epoch: 3 loops
- optimizer: Adagrad

#### Result
- Test acc: 85% with filter_num = 30, kernel_size = 5
- Test acc: 83% with filter_num = 20, kernel_size = 5
- Test acc: 87% with filter_num = 50, kernel_size = 5
- Test acc: 84% with filter_num = 20, kernel_size = 3
- Test acc: 84% with filter_num = 20, kernel_size = 5, 3 C&MP layers
- Test acc: 88% with filter_num = 50, 35, 20, kernel_size = 3, 3 C&MP layers, leaky_ReLU
- Test acc: 89% with filter_num = 50, 35, 20, kernel_size = 3, 3 C&MP layers, leaky_ReLU
