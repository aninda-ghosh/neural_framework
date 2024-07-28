# Backward Propagation in a Fully Connected Layer

When performing backpropagation, we need to calculate the following gradients for a Fully Connected (FC) layer:
- The derivative of the error with respect to the parameters (∂E/∂W and ∂E/∂B).
- The derivative of the error with respect to the input (∂E/∂X).

### Notations:
- **E**: Error
- **Y**: Output
- **X**: Input
- **W**: Weights
- **B**: Biases

## Gradients Calculation

### 1. Derivative of the Error with Respect to the Weights (∂E/∂W)

The weight matrix **W** has a size of **i x j**, where **i** is the number of input neurons and **j** is the number of output neurons. We need to calculate a gradient for each weight.

Using the chain rule:

    ∂E/∂W = (∂E/∂Y) * (∂Y/∂W)

Since **Y = X * W + B**:

    ∂Y/∂W = X^T

Therefore:

    ∂E/∂W = X^T * (∂E/∂Y)

### 2. Derivative of the Error with Respect to the Biases (∂E/∂B)

The bias matrix **B** has a size of **1 x j**. We need to calculate a gradient for each bias.

Using the chain rule:

    ∂E/∂B = (∂E/∂Y) * (∂Y/∂B)

Since **Y = X * W + B**:

    ∂Y/∂B = 1

Therefore:

    ∂E/∂B = ∂E/∂Y

### 3. Derivative of the Error with Respect to the Input (∂E/∂X)

The gradient **∂E/∂X** will be used as **∂E/∂Y** for the previous layer.

Using the chain rule:

    ∂E/∂X = (∂E/∂Y) * (∂Y/∂X)

Since **Y = X * W + B**:

    ∂Y/∂X = W^T

Therefore:

    ∂E/∂X = (∂E/∂Y) * W^T

## Summary of Formulas

1. **Gradient with respect to weights**:
    
       ∂E/∂W = X^T * (∂E/∂Y)

2. **Gradient with respect to biases**:
    
       ∂E/∂B = ∂E/∂Y

3. **Gradient with respect to inputs**:
    
       ∂E/∂X = (∂E/∂Y) * W^T
