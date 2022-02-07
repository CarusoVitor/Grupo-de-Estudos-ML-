import numpy as np
from typing import Callable
from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown
from activation_functions import *
from nnfs.datasets import spiral_data
from torchvision import transforms, datasets, utils


console = Console()


# =========
#   LAYERS
# =========


class Layer:
    ''' Fully-Connected Neural Network Layer '''
    def __init__(self, n_of_inputs, n_of_neurons, activation: Callable[[np.ndarray], np.ndarray], activation_prime, bias=0.0, name='Layer', log=False):
        self.n_of_inputs = n_of_inputs
        self.n_of_neurons = n_of_neurons
        self.weights = 0.1 * np.random.randn(n_of_inputs, n_of_neurons)
        self.bias = np.ones((1, n_of_neurons)) * bias
        self.activation = activation
        self.activation_prime = activation_prime
        self.name = name
        self.layer_inputs = None
        self.output = None
        if log: self.log()

    def log(self):
        # Visualização
        table = Table(show_header=True, header_style="bold magenta")
        row_str = []
        for i in range(self.n_of_neurons):
            col_str = 'Pesos Neurônio #' + str(i)
            table.add_column(col_str, width=20)
            row_str.append(str(self.weights.T[i]))

        table.add_row(*row_str)

        md = Markdown(f'# {self.name}')
        console.print(md)
        console.print(f"Matriz de pesos inicializados ({self.n_of_neurons} neurons, {self.n_of_inputs} inputs)")
        console.print(table, '\n')

    def feed_forward(self, x, log=False):
        # Visualização
        self.layer_inputs = x
        dot_product = x @ self.weights
        Z = dot_product + self.bias
        if log:
            console.print(f"Calculando dot de\n{x}\npor\n{self.weights}")
            console.print(f"Dot product:\n{dot_product}\nZ (Dot + bias = {self.bias}):\n{Z}\n")
            console.print(f"A(L) (sigmoid):\n{self.activation(Z)}")
        # print(Z)
        self.output = self.activation(Z) #A(L)
        return self.output

    def backward(self, derivative):

        # deriv: [d1, d2, d3]; dos neuronios da camada a direita
        # inp: [x1, x2, x3, x4] # valor recebido da camada imediatamente a esquerda
        dyhat_dw = self.layer_inputs
        # print(dyhat_dw.shape, derivative.shape)
        dL_dw = dyhat_dw.T @ derivative # (4, 1) @ (1, 3) = (4, 3)
        dyhat_a_1 = self.weights
        # print(derivative.shape, self.weights.shape)
        # print(self.layer_inputs.shape)
        # print(dyhat_a_1)
        # print(derivative.T.shape, dyhat_a_1.shape)
        dL_a_1 = derivative @ dyhat_a_1.T # (1, 3) @ (3, 4) = (1, 4)
        # print(self.weights.shape, dL_dw.shape)
        self.weights -= dL_dw * 0.03
        # print(dL_a_1.shape)
        # print(dL_a_1)
        # dL_a_1 = np.sum(dL_a_1, axis=0)
        # print(dL_a_1)
        return dL_a_1


class NeuralNetwork:
    def __init__(self, input_size):
        self.layers = []
        self.input_size = input_size

    def forward(self, x):
        for layer in self.layers:
            # print(x.shape)
            x = layer.feed_forward(x)
        # print(x.shape)
        return x

    def backward(self, loss_derivative):
        derivative = loss_derivative
        # print(f'Ds> {derivative} {derivative.shape}')
        # last_layer = None
        for layer in reversed(self.layers):
            derivative = layer.backward(derivative)
            # print(f'Ds> {derivative} {derivative.shape}')

    # Dado um número de saída adiciona uma camada ao fim da rede neural
    #   Ex: nn = NeuralNetwork(...)
    #       nn.append_layer(...)
    #       nn.append_layer(...)
    #       ...
    #

    # output_num
    def append_layer(self, output_number: int, bias: float, activation: Callable[[np.ndarray], np.ndarray], activation_prime):
        if len(self.layers) == 0:
            new_layer_input = self.input_size
        else:
            new_layer_input = self.layers[-1].n_of_neurons

        self.layers.append(Layer(new_layer_input, output_number, activation, activation_prime, bias))


def MSELoss(y_hat, y):
    return np.sum((y_hat - y)**2)/len(y_hat)


def MSELoss_prime(y_hat, y):
    return 2*(y_hat - y)


# (3, 4) (4, 1)
nn = NeuralNetwork(3)
nn.append_layer(4, bias=1, activation=sigmoid, activation_prime=sigmoid_prime)
nn.append_layer(3, bias=1, activation=sigmoid, activation_prime=sigmoid_prime)

X = np.array([[1, 2, 3]])
y = np.array([[1, 0, 0]])

for epoch in range(30):
    predicted = nn.forward(X)
    loss = MSELoss(predicted, y)
    loss_derivative = MSELoss_prime(predicted, y)
    nn.backward(loss_derivative)
    print(loss)

# Expected type '(ndarray) -> ndarray', got 'Type[Sigmoid]' instead

# ================ #
# Data Preparation #
# ================ #
#
# Setting up the transformations needed for the digits images
transform = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

# Downloading MNIST dataset
# data_path='/data/mnist'
# mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
# num_classes = 10  # MNIST has 10 output classes
# print(f"The size of mnist_train is {len(mnist_train)}")
#
# # Defining a subset with N=10 items
# subset = 10
# utils.data_subset(mnist_train, subset)
# print(f"The size of mnist_train is {len(mnist_train)}")