import matplotlib.pyplot as plt

"""
Neural Network class for training and prediction
"""
class Network:
    def __init__(self, loss):
        """
        Initializes the Network with a loss function.
        
        :param loss: The loss function to be used for training.
        """
        self.layers = []
        self.loss = loss
        self.epoch_errors = []

    def add(self, layer):
        """
        Adds a layer to the network.
        
        :param layer: The layer to be added to the network.
        """
        self.layers.append(layer)
    
    def __repr__(self):
        """
        Provides a string representation of the network, showing the layers and their sizes.
        
        :return: String representation of the network.
        """
        layer_descriptions = []
        for i, layer in enumerate(self.layers):
            input_size = getattr(layer, 'input_size', 'Unknown')
            output_size = getattr(layer, 'output_size', 'Unknown')
            layer_info = f"  ({i}): {layer.__class__.__name__} (input: {input_size}, output: {output_size})"
            layer_descriptions.append(layer_info)
        return f"{self.__class__.__name__}(\n" + "\n".join(layer_descriptions) + f"\n  (loss): {self.loss.__class__.__name__}\n)"

    def fit(self, x_train, y_train, epochs, learning_rate):
        """
        Trains the network using the given training data.
        
        :param x_train: Training input data.
        :param y_train: Training target data.
        :param epochs: Number of epochs to train the network.
        :param learning_rate: Learning rate for updating the parameters.
        """
        samples = len(x_train)
        self.epoch_errors = []
        for i in range(epochs):
            epoch_error = 0
            for j in range(samples):
                # Forward pass
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward(output)

                # Compute loss
                epoch_error += self.loss.forward(y_train[j], output)

                # Backward pass
                error_derivative = self.loss.backward(y_train[j], output)
                for layer in reversed(self.layers):
                    error_derivative = layer.backward(error_derivative, learning_rate)
                
            epoch_error /= samples # Average error
            print('epoch %d/%d   error=%f' % (i+1, epochs, epoch_error))
            self.epoch_errors.append(epoch_error)

    def plot(self):
        """
        Plots the training error over epochs.
        """
        plt.plot(range(1, len(self.epoch_errors) + 1), self.epoch_errors, marker='o')
        plt.title('Training Error Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.grid(True)
        plt.show()

    def predict(self, x_test):
        """
        Predicts the output for the given test data.
        
        :param x_test: Test input data.
        :return: Predicted output.
        """
        samples = len(x_test)
        res = []

        for i in range(samples):
            output = x_test[i]
            for layer in self.layers:
                output = layer.forward(output)
            res.append(output)
        
        return res
