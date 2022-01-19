from activation_functions import *
from loss_function import *
from layer_dense import *
from loss_function import *
from nnfs.datasets import spiral_data

console = Console()


# ========
#   TESTS
# ========


# Exemplo de feedforward em uma camada
# example_layer1 = DenseLayer(3, 2, relu)           # Camada com bias = 0
# example_layer2 = DenseLayer(3, 2, relu, bias=1)   # Camada com bias = 1
# input1 = np.array([[1, 2, 3]])              # 1 batch
# input2 = np.array([[1, 2, 3], [4, 5, 6]])   # 2 batches
#
# example_layer1.feed_forward(input1, True) # Exemplo de layer recebendo um batch
# example_layer2.feed_forward(input2, True) # Exemplo de layer recebendo dois batches




# Instânciando camadas densas
layer1 = DenseLayer(2, 3, relu, name='Layer 1', log=True)
layer2 = DenseLayer(3, 3, relu, name='Layer 2', log=True)
layer3 = DenseLayer(3, 3, softmax, name='Layer 3', log=True)


# Coletadando dados
X, y = spiral_data(samples=100, classes=3)


# Aplicando o Forward Pass nas camadas



layer1.feed_forward(X)
layer2.feed_forward(layer1.output)
prediction_confidences = layer3.feed_forward(layer2.output)
print(prediction_confidences[:5])

loss_fn = LossCategoricalCrossEntropy()
loss = loss_fn.calculate(prediction_confidences, y)



# Quase uma rede neural
print(f"Input -> Layer1:\n{layer1.feed_forward(X[:5])}\n-----------------")
print(f"Layer1 -> Layer2:\n{layer2.feed_forward(layer1.output)[:5]}\n-----------------")
print(f"Layer2 -> Output:\n{prediction_confidences[:5]}")
print(f"Loss: {loss}")







