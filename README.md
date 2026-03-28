# End-to-End MNIST Digit Classification with TensorBoard Monitoring

## Overview
This project is an end-to-end Deep Learning workflow for classifying handwritten digits (0-9) using the classic MNIST dataset. The primary objective is to demonstrate iterative model optimization—starting with a baseline Single-Layer Neural Network and upgrading to a Multi-Layer Feedforward Neural Network to achieve near-perfect predictive accuracy. The project features comprehensive performance monitoring using TensorBoard and visual misclassification analysis via Seaborn heatmaps.

## Tech Stack
* **Language:** Python 3.13
* **Deep Learning Framework:** TensorFlow, Keras
* **Monitoring & Visualization:** TensorBoard, Matplotlib, Seaborn
* **Data Manipulation:** NumPy

## Dataset
* **Source:** MNIST (Modified National Institute of Standards and Technology) dataset, loaded via `keras.datasets.mnist`.
* **Size:** 60,000 training images and 10,000 testing images.
* **Format:** 28x28 pixel grayscale images, flattened into 1D arrays of 784 elements.

## Methodology & Workflow

### 1. Data Preprocessing & Pipeline
* **Normalization:** Scaled pixel values from a range of 0-255 to 0-1 to ensure stable gradient descent and faster model convergence.
* **Dimensionality Reshaping:** Flattened the 2D matrices (28x28) into 1D arrays (784) for ingestion into the dense neural network layers.

### 2. Model Architecture & Iteration
* **Baseline Model:** Engineered a preliminary single-layer model with a 10-neuron `softmax` output, achieving an initial baseline test accuracy of ~87.5%.
* **Optimized Deep Neural Network:** Upgraded the architecture by introducing a hidden `Dense` layer (100 neurons) with a `ReLU` (Rectified Linear Unit) activation function to capture non-linear, complex image patterns. Compiled using the `adam` optimizer and `sparse_categorical_crossentropy` loss.

### 3. Monitoring & Evaluation
* **TensorBoard Integration:** Implemented Keras callbacks to stream training metrics (loss and accuracy scalars) directly into TensorBoard for real-time visualization and to monitor for overfitting across epochs.
* **Granular Verification:** Replaced standard accuracy metrics with a robust Confusion Matrix heatmap (via Seaborn) to cross-reference predictions against truth labels, identifying specific edge-case confusions (e.g., distinguishing handwritten 4s from 9s).

### 4. Custom Inference Pipeline
* Developed a standalone prediction script capable of ingesting external, user-generated images. The pipeline handles target resizing, grayscale conversion, color inversion, and normalization before passing the novel data to the trained model for real-world testing.

## Key Results

| Model Architecture                  | Training Accuracy (10 Epochs) | Test Accuracy |
| :---                                | :---                          | :---          |
| **Baseline** (No Hidden Layers)     | ~89.1%                        | 87.5%         |
| **Optimized** (1 Hidden Layer, ReLU)| 99.8%                         | ~97.5%        |

## How to Run

1. Clone the repository to your local machine.
2. Ensure you have the required dependencies installed in your virtual environment:
   ```bash
   pip install tensorflow numpy matplotlib seaborn tensorboard jupyter
3. Run the Jupyter Notebook to train and evaluate the model.

4. To view the training graphs and metrics, launch TensorBoard via your terminal:
   ```bash
   tensorboard --logdir logs/fit