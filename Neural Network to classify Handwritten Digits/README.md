**TASK3**
#

## Simple MNIST Neural Network

This repository contains a simple implementation of a two-layer neural network built from scratch to recognize handwritten digits from the MNIST dataset using python and numpy.

### Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

### Features

- Implements a two-layer neural network.
- Trains on the MNIST digit recognition dataset.
- Includes functions for forward propagation, backward propagation, and parameter updates.
- Visualizes predictions alongside actual labels.

### Getting Started

To get a local copy of this project up and running, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Isaadqurashi/DEP/tree/main/Neural%20Network%20to%20classify%20Handwritten%20Digits.git
   ```
2. Navigate to the project directory:
   ```bash
   cd simple-mnist-nn
   ```
3. Install the required packages:
   ```bash
   pip install numpy pandas matplotlib
   ```

### Usage

To run the neural network, execute the following command in your terminal:

```bash
python mnist_nn.py
```

This will train the model on the MNIST dataset and display the accuracy of predictions at various iterations.

### Training the Model

The model is trained using gradient descent. The main steps involved are:

- **Initialization**: Randomly initializes weights and biases.
- **Forward Propagation**: Computes the output of the network given the input data.
- **Backward Propagation**: Calculates gradients and updates parameters.
- **Iterations**: Repeats the forward and backward propagation for a specified number of iterations.

### Results

The model achieves approximately **85% accuracy** on the training set. During training, the accuracy is printed at every 10 iterations. The final accuracy on the validation set can be checked at the end of training.

### Example Predictions

The model can make predictions on individual images from the dataset. Here are a few examples:

- Prediction: 0, Label: 0
- Prediction: 9, Label: 9
- Prediction: 8, Label: 9
- Prediction: 2, Label: 2

### Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

### License

This project is licensed under the MIT License. See the LICENSE file for more details.

---