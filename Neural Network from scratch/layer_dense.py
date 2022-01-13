import numpy as np
from typing import Callable
from rich.console import Console
from rich.table import Table
console = Console()

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


# =========
#   LAYERS
# =========


class DenseLayer:
    ''' Fully-Connected Neural Network Layer '''
    def __init__(self, n_of_inputs, n_of_neurons, activation: Callable[[np.ndarray], np.ndarray], bias=0, log=False):
        self.n_of_inputs = n_of_inputs
        self.n_of_neurons = n_of_neurons
        self.weights = 0.1 * np.random.randn(n_of_inputs, n_of_neurons)
        self.bias = np.ones((1, n_of_neurons)) * bias
        self.activation = activation

        # Visualização
        if log:
            console.print(f"Matriz de pesos inicializados ({n_of_inputs} inputs, {n_of_neurons} neurons)\n{self.weights}\n")
            table = Table(show_header=True, header_style="bold magenta")
            for i in range(n_of_neurons):
                my_str = 'Neurônio '+ str(i)
                table.add_co
        table.add_row(str(self.weights[j][i]))

            console.print(table)

    def feed_forward(self, input, log=False):
        # Visualização
        dot_product = np.dot(input, self.weights)
        Z = dot_product + self.bias
        if log:
            console.print(f"Calculando dot de\n{input}\npor\n{self.weights}")
            console.print(f"Dot product:\n{dot_product}\nZ (Dot + bias = {self.bias}):\n{Z}\n")
            console.print(f"A(L) (sigmoid):\n{self.activation(Z)}")
        # print(Z)
        return self.activation(Z) #A(L)


# ===============
# LOSS FUNCTIONS
# ===============


def categorical_loss_entropy(x: np.ndarray, y: np.ndarray):
    log_part = np.log(x) # [log(0.7), log(0.1), log(0.2)]
    mult_part = log_part * y # [log(0.7), log(0.1), log(0.2)]) * [1, 0, 0]
    loss = -np.sum(mult_part) # -sum(log(0.7) * 1, log(0.1) * 0, log(0.2) * 0])
    return loss

# ([log(0.7), log(0.1), log(0.2)]) * [1, 0, 0] = -sum(log(0.7) * 1, log(0.1) * 0, log(0.2) * 0])




# Exemplo de feedforward em uma camada
# example_layer1 = DenseLayer(3, 2, relu)           # Camada com bias = 0
# example_layer2 = DenseLayer(3, 2, relu, bias=1)   # Camada com bias = 1
# input1 = np.array([[1, 2, 3]])              # 1 batch
# input2 = np.array([[1, 2, 3], [4, 5, 6]])   # 2 batches
#
# example_layer1.feed_forward(input1, True) # Exemplo de layer recebendo um batch
# example_layer2.feed_forward(input2, True) # Exemplo de layer recebendo dois batches

# Exemplo de feedforward de duas camadas consecutivas
layer1 = DenseLayer(3, 2, relu, log=True)
layer2 = DenseLayer(2, 4, relu, log=True)
layer3 = DenseLayer(4, 4, softmax, log=True)
test_input = np.array([[1, 2, 3], [4, 5, 6]])

# test_input = np.array([1, 2, 3])

# Quase uma rede neural
output_layer1 = layer1.feed_forward(test_input)
print(f"Input -> Layer1:\n{output_layer1}\n-----------------")
output_layer2 = layer2.feed_forward(output_layer1)
print(f"Layer1 -> Layer2:\n{output_layer2}\n-----------------")
output_layer3 = layer3.feed_forward(output_layer2)
print(f"Layer2 -> Layer3 (Output layer):\n{output_layer3}")
# test = np.array([-1,-2,1,4,-5])
# print(relu(test))

x0 = np.array([0.99, 0.0025, 0.0025, 0.0025, 0.0025])
x1 = np.array([0.0025, 0.99, 0.0025, 0.0025, 0.0025])
y0 = np.array([1, 0, 0, 0, 0])
print(categorical_loss_entropy(x0, y0))
print(categorical_loss_entropy(x1, y0))