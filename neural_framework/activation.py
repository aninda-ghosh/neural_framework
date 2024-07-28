import numpy as np

# Hyperbolic Tangent (tanh)
# The tanh function maps any real-valued number to the range [-1, 1].
def tanh(x):
    return np.tanh(x)

# Derivative of the tanh function.
# The derivative of tanh is: 1 - tanh(x)^2
def tanh_prime(x):
    return 1 - np.tanh(x)**2


# Sigmoid
# The sigmoid function maps any real-valued number to the range [0, 1].
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function.
# The derivative of sigmoid is: sigmoid(x) * (1 - sigmoid(x))
def sigmoid_prime(x):
    return x * (1 - x)


# Rectified Linear Unit (ReLU)
# The ReLU function outputs the input directly if it is positive; otherwise, it outputs zero.
def relu(x):
    return np.maximum(0, x)

# Derivative of the ReLU function.
# The derivative of ReLU is: 1 if x > 0, else 0
def relu_prime(x):
    return np.where(x > 0, 1, 0)
