## Model Specs and Results

### Mul_layer_CNN Model Structures
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
- epoch: 10 loops
- optimizer: Adagrad

#### Result
- Test set acc: 91%, avg loss: 0.0026, 10 epochs

### 4Layer_model Structures
- Conv layers: 2/3
	* conv1: size 1, 50
	* conv2: size 50, 35
	* conv3: size 35, 20 
- MaxPool layers: 2
	* MaxPool: 2
- Activation: leaky_ReLU
- Dense layers: 2:
	* Dense(875, 200)
	* Dense(200, 10)

#### Training tools/parameters
- Loss: CrossEntropyLoss
- epoch: 5 loops
- optimizer: Adagrad

#### Result
- Test acc: 89% with filter_num = 50, 35, 20, kernel_size = 3, 3 C&MP layers, leaky_ReLU
- 
- Test acc: 85% with filter_num = 30, kernel_size = 5
- Test acc: 83% with filter_num = 20, kernel_size = 5
- Test acc: 87% with filter_num = 50, kernel_size = 5
- Test acc: 84% with filter_num = 20, kernel_size = 3
- Test acc: 84% with filter_num = 20, kernel_size = 5, 3 C&MP layers
- Test acc: 88% with filter_num = 50, 35, 20, kernel_size = 3, 3 C&MP layers, leaky_ReLU

