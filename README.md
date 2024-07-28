# Neural Network From Scratch

## Description

This is a basic neural framework to train a Deep Neural Network with an arbitrary input and output size. 

## Capabilities and Caveats

### Capabilities

- Users can choose between two activation functions. Other activation functions (e.g Leaky Relu, Gelu) can be easily integrated in the future.
    - Sigmoid
    - Relu
    - Tanh
- Loss Function provided. Other loss functions (e.g Mean Absolute Error, Cross Entropy Error) can be easily integrated in the future.
    - Mean Square Error
- Supports vanilla Gradient Descent.

### Caveats

- Don't have support for Autograd in this version.
- Gradient accumulation is limited in nature.
- Batched backpropagataion is not implemented yet.
- Optimizers like (SGD, ADAM and RMS Prop) are still not implemented in the minimal framework.

## Installation

- Clone the repository containing the framework code.
  ```
  git clone git@github.com:aninda-ghosh/neural_framework.git
  ```

- Create a Conda environment
  ```
  conda env create -f env.yml
  ```

- Install the framework in the created conda environment
  ```
  conda activate neural_framework
  pip install -e .
  ```

## References

- https://d2l.ai/chapter_builders-guide/index.html