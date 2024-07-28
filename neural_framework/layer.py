import numpy as np

"""
Base class that will be derived by other layers to compute the forward and backward implementation
"""
class Layer:
    """
    Init will initialize the inputs and outputs as None.
    """
    def __init__(self):
        self.input = None
        self.output = None
        self.input_size = None
        self.output_size = None
    
    """
    Computes the output Y given an input X
    """
    def forward(self, input):
        raise NotImplementedError
    
    """
    Computes DE/Dx for a given DE/Dy (Updates Params if any)
    """
    def backward(self, output_error, learning_rate):
        raise NotImplementedError



"""
Fully Connected Layer (Completely Linear Layer)
"""
class FullyConnectedLayer(Layer):
    """
    Initializes the FullyConnectedLayer with random weights and biases.
    
    :param input_size: Size of the input vector.
    :param output_size: Size of the output vector.
    """
    def __init__(self, input_size, output_size):
        # Weight and bias initialization
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    """
    Computes the output Y given an input X.
    
    :param input: Input data.
    :return: Output of the fully connected layer.
    """
    def forward(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    """
    Computes DE/Dx for a given DE/Dy and updates the parameters.
    
    :param output_error: Gradient of the loss with respect to the output.
    :param learning_rate: Learning rate for updating parameters.
    :return: Gradient of the loss with respect to the input.
    """
    def backward(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        bias_error = output_error

        # Update weights and bias
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error

        return input_error

"""
Activation Layer (Any defined activation can be passed here)
"""
class ActivationLayer(Layer):
    """
    Initializes the ActivationLayer with the given activation function and its derivative.
    
    :param activation: Activation function.
    :param activation_prime: Derivative of the activation function.
    """
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
        self.input_size = 1
        self.output_size = 1
    
    """
    Computes the output Y given an input X.
    
    :param input: Input data.
    :return: Output after applying the activation function.
    """
    def forward(self, input):
        self.input = input
        self.output = self.activation(self.input)
        return self.output

    """
    Computes DE/Dx for a given DE/Dy.
    
    :param output_error: Gradient of the loss with respect to the output.
    :param learning_rate: Learning rate (not used in this layer).
    :return: Gradient of the loss with respect to the input.
    """
    def backward(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error
