import numpy as np

from activation_functions import *
from loss_function import *
from layer_dense import *
from loss_function import *
from nnfs.datasets import spiral_data
from typing import List

class NeuralNet:
    def __init__(self, input_size, output_size, hidden_size, n_hidden_layers=2):
        self.input_size = input_size
        self.output_size = output_size

        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers

        self.input_layer = DenseLayer(input_size, hidden_size, relu)
        self.hidden_layer = DenseLayer(hidden_size, hidden_size, relu)
        self.output_layer = DenseLayer(hidden_size, output_size, softmax)

        # Lista de layers. O ultimo elemento corresponde à última camada (e portanto sua saída é a da rede)
        # self.layers: List[DenseLayer] = []

    def forward(self, X):
        X = self.input_layer.feed_forward(X)
        for n in range(self.n_hidden_layers):
            X = self.hidden_layer.feed_forward(X)
        output = self.output_layer.feed_forward(X)
        return output

    # Forward generico
    # def forward(self, cur_input: np.ndarray):
    #     cur_output = cur_input
    #     for layer in self.layers:
    #         cur_output = layer.feed_forward(cur_output)
    #     return cur_output

    # Dado um número de saída adiciona uma camada ao fim da rede neural
    #   Ex: nn = NeuralNetwork(...)
    #       nn.append_layer(...)
    #       nn.append_layer(...)
    #       ...
    #
    # def append_layer(self, output_number: int, bias: float, activation: Callable[[np.ndarray], np.ndarray]):
    #     if len(self.layers) == 0:
    #         new_layer_input = self.input_size
    #     else:
    #         new_layer_input = self.layers[-1].n_of_neurons
    #
    #     self.layers.append(DenseLayer(new_layer_input, output_number, activation, bias))
