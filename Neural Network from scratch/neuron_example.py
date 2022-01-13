import numpy as np
# import torch


# if torch.cuda.is_available():
#     device = torch.device("cuda")

# Output calculado na mão
inputs = [1.2, 5.1, 2.1]
# inputs = [[1, 2, 3, 2.5],
#           [2, 5, -1, 2],
#           [-1.5, 2.7, 3.3, -0.8]]

weights = [3.1, 2.1, 8.7]
bias = 3

# Output com numpy
inputs_np = np.array(inputs)            # Matriz de input
weights_np = np.array(weights)          # Matriz dos pesos


class Neuron:
    def __init__(self, weights:np.array, bias:int):
        self.weights = weights
        self.bias = bias

    def output(self, inputs):
        return np.dot(inputs, self.weights) + self.bias

    def activation_function(self, inputs):
        pass


# output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias
neuronio = Neuron(weights_np, bias)
print(neuronio.output(inputs_np))

#                    [ 3.1 ]
# [1.2  5.1  2.1] *  [ 2.1 ]   = x11w11 + x12w21 + x13w31
#                    [ 8.7 ]

# print(inputs_np.shape, inputs_weights.shape)


# print(output)
# print(output_np)