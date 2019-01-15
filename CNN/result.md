## Models and thier results

* Using FashionMNIST data

### CNN
 	* Loss: CrossEntropyLoss
 	* Conv layers: 2
		* conv1(1, 70, kernel=5)
		* conv2(70, 20, kernel=3)
	* MaxPool 2
		* mp(2)
	* Activation: leaky_ReLU
	* Dense 1
		* dense(500, 10)
	* Result: test set acc: 87%, with 4 loops on training set

### v2.0 mul_layer
	- Conv layers: 2
		* conv1: size 1, 64 
		* conv2: size 64, 64 
		* conv3: size 64, 32 
	- MaxPool layers: 2
		* MaxPool: 2
	- Activation: leaky_ReLU
	- loss: CrossEntropyLoss
	- Dense layers: 2:
		* Dense(1600, 512)
		* Dense(512, 128)
		* Dense(128, 10)
	- Result: test acc: 91%, avg loss: 0.0026, 10 epochs

### 4Layer_CNN
	* Loss: CrossEntropy
	* Conv 4
		* conv1(1, 50, kernel=5)
		* conv2(50, 40, kernel=4)
		* conv3(40, 40, kernel=3)
		* conv4(40, 30, kernel=3)
	* MaxPool 3
		* mp1(3, stride=1)
		* mp2(2)
		* mp3(2, stride=1)
	* dense 2
		* dense1(875, 200)
		* dense2(400, 10)
	* Result: test acc: 90%, avg loss: 0.0027 , 11 epochs



