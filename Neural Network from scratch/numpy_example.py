import numpy as np

# inputs = [1.2, 5.1, 2.1]
# weights = [3.1, 2.1, 8.7]
# bias = 3
# output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias
#
# # Output com numpy
# inputs_np = np.array(inputs)
# # print(inputs_np)
# inputs_weights = np.array(weights)
# output_np = np.dot(inputs_weights, inputs_np) + bias
# inputs_np @ output_np
# print(output_np)
# np.random.randn(4, 3)
# np.zeros((1, 5))
# # print(inputs_np)
# # print(inputs_np.shape, inputs_weights.shape)


def calculadora(n1, n2, operacao):
    return operacao(n1, n2)


def soma(n1, n2):
    return n1 + n2


print(calculadora(3, 2, soma))