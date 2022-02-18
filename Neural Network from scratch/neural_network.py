from typing import Callable
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown
from activation_functions import *
from torchvision import transforms, datasets


console = Console()
# =========
#   LAYERS
# =========

class Layer:
    ''' Fully-Connected Neural Network Layer '''
    def __init__(self, n_of_inputs, n_of_neurons, activation: Callable[[np.ndarray], np.ndarray], activation_prime, bias=0.0, name='Layer', log=False):
        self.n_of_inputs = n_of_inputs
        self.n_of_neurons = n_of_neurons
        self.weights = np.random.uniform(-1, 1, (n_of_inputs, n_of_neurons))
        self.weight_derivatives: int = 0
        self.bias_derivatives: int = 0
        self.bias = np.ones((1, n_of_neurons)) * bias
        self.activation = activation
        self.activation_prime = activation_prime
        self.name = name
        self.layer_inputs = None
        self.z = None
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
        self.z = dot_product + self.bias
        if log:
            console.print(f"Calculando dot de\n{x}\npor\n{self.weights}")
            console.print(f"Dot product:\n{dot_product}\nZ (Dot + bias = {self.bias}):\n{self.z}\n")
            console.print(f"A(L) (sigmoid):\n{self.activation(self.z)}")
        # print(Z)
        output = self.activation(self.z) #A(L)
        return output

    def backward(self, derivative):
        # deriv: [d1, d2, d3]; dos neuronios da camada a direita
        # inp: [x1, x2, x3, x4] # valor recebido da camada imediatamente a esquerda
        dyhat_db = sigmoid_prime(self.z)
        dL_db = derivative * dyhat_db
        dyhat_dw = self.layer_inputs
        dL_dw = dyhat_dw.T @ (derivative * sigmoid_prime(self.z)) # (m1, n1) @ (m2, n2) = (m1, n2)
        dyhat_a_1 = self.weights
        dL_a_1 = (derivative * sigmoid_prime(self.z)) @ dyhat_a_1.T # (1, 3) @ (3, 4) = (1, 4)
        self.weight_derivatives = dL_dw
        self.bias_derivatives = dL_db
        return dL_a_1


class NeuralNetwork:
    def __init__(self, input_size, lr):
        self.layers = []
        self.input_size = input_size
        self.lr = lr

    def forward(self, x):
        for layer in self.layers:
            x = layer.feed_forward(x)
        return x

    def backward(self, loss_derivative):
        derivative = loss_derivative
        # print(f'Ds> {derivative} {derivative.shape}')
        # last_layer = None
        for layer in reversed(self.layers):
            derivative = layer.backward(derivative)
            # print(f'Ds> {derivative} {derivative.shape}')

        for layer in self.layers:
            layer.weights -= layer.weight_derivatives * self.lr
            layer.bias -= layer.bias_derivatives * self.lr

    def __call__(self, inputs):
        return self.forward(inputs)

    def append_layer(self, output_number: int, bias: float, activation: Callable[[np.ndarray], np.ndarray], activation_prime):
        """
        Dado um número de saída adiciona uma camada ao fim da rede neural
        Ex: nn = NeuralNetwork(...)
          nn.append_layer(...)
          nn.append_layer(...)
          ...
        """
        if len(self.layers) == 0:
            new_layer_input = self.input_size
        else:
            new_layer_input = self.layers[-1].n_of_neurons

        self.layers.append(Layer(new_layer_input, output_number, activation, activation_prime, bias))


def MSELoss(y_hat, y):
    return np.sum((y_hat - y)**2)/len(y_hat)


def MSELoss_prime(y_hat, y):
    return 2*(y_hat - y)


# Cria vetor one hot mnist
def one_hot(value: int):
    one_hot_vec = np.zeros((1, 10))
    one_hot_vec[0][value] = 1
    return one_hot_vec


if __name__ == "__main__":
    # ================ #
    # Data Preparation #
    # ================ #
    #
    # Setting up the transformations needed for the digits images
    transform = transforms.Compose([
                transforms.Resize((28,28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5)
    ])

    # Downloading MNIST dataset
    data_path='/data/mnist'
    mnist_train = datasets.MNIST(data_path, train=True, transform=transform)
    mnist_test = datasets.MNIST(data_path, train=False, transform=transform)
    num_classes = 10  # MNIST has 10 output classes
    train_set = mnist_train.data.numpy()/255
    train_targets = [one_hot(t) for t in mnist_train.targets]
    test_set = mnist_test.data.numpy()/255
    test_targets = [one_hot(t) for t in mnist_test.targets]
    print(f"Size of train set is {len(train_set)}")
    print(f"Size of test set is {len(test_set)}")
    lr = 0.003
    nn = NeuralNetwork(28*28, lr)
    nn.append_layer(64, bias=1, activation=sigmoid, activation_prime=sigmoid_prime)
    nn.append_layer(64, bias=1, activation=sigmoid, activation_prime=sigmoid_prime)
    nn.append_layer(10, bias=1, activation=sigmoid, activation_prime=sigmoid_prime)

    for epoch in range(10):
        total_loss = 0
        for i, (x, y) in enumerate(zip(train_set, train_targets)):
            y_hat = nn(x.flatten().reshape(1,-1))
            loss = MSELoss(y_hat, y)
            loss_derivative = MSELoss_prime(y_hat, y)
            nn.backward(loss_derivative)
            total_loss += loss
            if i % 20000 == 0 and i > 0:
                print(f"[{epoch+1}, {i+1:5d}] Accumulated Loss: {total_loss/(20000 * i)}")

        print(f"Loss: {total_loss / len(train_set)} - Epoch: {epoch + 1}")

        # validate
        hits = 0
        for x, y in zip(test_set, test_targets):
            y_hat = nn(x.flatten().reshape(1, -1))
            if np.argmax(y_hat) == np.argmax(y):
                hits += 1
        print(f'\nEpoch Accuracy: {hits/len(test_set)* 100}%\n')


def guess():
    n = np.random.randint(0, len(test_set))
    predicted = nn.forward(test_set[n].flatten().reshape(1, -1))
    actual = test_targets[n]
    print(f"Actual number: {np.argmax(actual)} - Predicted number: {np.argmax(predicted)}")