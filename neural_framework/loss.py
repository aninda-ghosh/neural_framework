import numpy as np

"""
Base class for loss functions that will be derived by specific loss implementations.
"""
class Loss:
    def __init__(self):
        pass

    """
    Computes the loss value given the true labels and predicted labels.
    
    :param y_true: The true labels.
    :param y_pred: The predicted labels.
    :return: The computed loss value.
    """
    def forward(y_true, y_pred):
        raise NotImplementedError

    """
    Computes the gradient of the loss with respect to the predicted labels.
    
    :param y_true: The true labels.
    :param y_pred: The predicted labels.
    :return: The gradient of the loss with respect to the predicted labels.
    """
    def backward(y_true, y_pred):
        raise NotImplementedError


"""
Mean Squared Error (MSE) Loss Function
"""
class MSE(Loss):
    def __init__(self):
        pass

    """
    Computes the Mean Squared Error loss given the true labels and predicted labels.
    
    :param y_true: The true labels.
    :param y_pred: The predicted labels.
    :return: The Mean Squared Error loss value.
    """
    def forward(self, y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    """
    Computes the gradient of the Mean Squared Error loss with respect to the predicted labels.
    
    :param y_true: The true labels.
    :param y_pred: The predicted labels.
    :return: The gradient of the loss with respect to the predicted labels.
    """
    def backward(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size
