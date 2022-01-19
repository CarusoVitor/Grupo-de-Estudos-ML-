import numpy as np
from typing import Callable
from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown
from activation_functions import *

console = Console()


# =========
#   LAYERS
# =========


class DenseLayer:
    ''' Fully-Connected Neural Network Layer '''
    def __init__(self, n_of_inputs, n_of_neurons, activation: Callable[[np.ndarray], np.ndarray], bias=0, name='Layer', log=False):
        self.n_of_inputs = n_of_inputs
        self.n_of_neurons = n_of_neurons
        self.weights = 0.1 * np.random.randn(n_of_inputs, n_of_neurons)
        self.bias = np.ones((1, n_of_neurons)) * bias
        self.activation = activation
        self.name = name
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

    def feed_forward(self, input, log=False):
        # Visualização
        dot_product = np.dot(input, self.weights)
        Z = dot_product + self.bias
        if log:
            console.print(f"Calculando dot de\n{input}\npor\n{self.weights}")
            console.print(f"Dot product:\n{dot_product}\nZ (Dot + bias = {self.bias}):\n{Z}\n")
            console.print(f"A(L) (sigmoid):\n{self.activation(Z)}")
        # print(Z)
        self.output = self.activation(Z) #A(L)
        return self.output
