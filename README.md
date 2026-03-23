# MNIST Digit Classification using TensorFlow & Keras

## Overview
This project is an end-to-end Deep Learning workflow for classifying handwritten digits (0-9) using the classic MNIST dataset. The primary objective of this project is to demonstrate iterative model improvement—starting with a baseline Single-Layer Neural Network and upgrading to a Multi-Layer Feedforward Neural Network to significantly increase predictive accuracy. Model performance is verified and analyzed using a Confusion Matrix.

## Tech Stack
* **Language:** Python 3
* **Deep Learning Framework:** TensorFlow, Keras
* **Data Manipulation:** NumPy
* **Data Visualization:** Matplotlib

## Dataset
* **Source:** MNIST (Modified National Institute of Standards and Technology) dataset, loaded directly via `keras.datasets.mnist`.
* **Size:** 60,000 training images and 10,000 testing images.
* **Format:** 28x28 pixel grayscale images, flattened into 1D arrays of 784 elements for the neural network.

## Methodology & Workflow

### 1. Data Preprocessing
* **Normalization:** Scaled pixel values from a range of 0-255 to 0-1 (e.g., `X / 255.0`). This ensures faster convergence during the gradient descent optimization process.
* **Reshaping:** Flattened the 2D images (28x28) into 1D arrays (784) to feed into the `Dense` layers of the neural network.

### 2. Baseline Model (Simple Neural Network)
* **Architecture:** A single output `Dense` layer with 10 neurons and a `softmax` activation function.
* **Compilation:** Used the `adam` optimizer and `sparse_categorical_crossentropy` loss function.
* **Result:** Achieved an initial test accuracy of **~87.5%**.

### 3. Improved Model (Deep Neural Network)
To increase the model's capacity to learn complex, non-linear patterns, the architecture was upgraded:
* **Architecture:** Added a hidden `Dense` layer with 100 neurons and a `ReLU` (Rectified Linear Unit) activation function, followed by the 10-neuron `softmax` output layer.
* **Result:** The addition of the hidden layer drastically improved model performance, pushing test accuracy to **~94.5%** and training accuracy to over 95%.

### 4. Evaluation & Verification
* **Confusion Matrix:** Utilized `tf.math.confusion_matrix` to cross-reference predicted labels against the actual truth labels. This granular evaluation step helped visualize exactly which digits the model was consistently misclassifying (e.g., confusing 4s and 9s), going beyond a simple accuracy metric.

## Key Results

| Model Architecture                  | Training Accuracy (10 Epochs) | Test Accuracy |
|                                :--- |                         :---  |          :--- |
| **Baseline** (No Hidden Layers)     |              ~89.1%           |         87.5% |
| **Improved** (1 Hidden Layer, ReLU) |              ~95.7%           |         94.5% |

## How to Run

1. Clone the repository.
2. Ensure you have the required dependencies installed:
   ```bash
   pip install tensorflow numpy matplotlib jupyter