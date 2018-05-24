### Models and thier results
- CNN
 * Loss: NLLLoss
 * output though: log_softmax
- CNN_1
 * Loss: CrossEntropy
 * output as it is
- v2.0 mul_layer
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
	- Result: test acc: 91%, avg loss: 0.0026, 10 epochs



