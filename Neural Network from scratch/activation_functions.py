import numpy as np


# ==============
#   ACTIVATIONS
# ==============


def sigmoid(x: np.ndarray):
    ''' Sigmoid Activation Function

        Output Interval: [0, 1]
    '''
    return 1 / (1 + np.exp(-x))


def relu(x: np.ndarray):
    ''' Rectified Linear Unit Activation Function

        Output Interval: [0, +inf)
    '''
    return np.maximum(0, x)


def softmax(x: np.ndarray):
    ''' SoftMax Activation Function

        Output Interval: [0, 1]
        Output Element-Wise Sum: 1
    '''
    y = np.exp(x)
    y_sum = np.sum(y, axis=1, keepdims=True)
    return y/y_sum