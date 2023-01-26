import numpy as np
import matplotlib.pyplot as plt


class NeuNet:
    def __init__(self, layers: list, learing_rate) -> None:
        self.layers = layers
        self.learing_rate = learing_rate

    def forward(self, x):
        x = x.reshape(x.shape[0], 1)
        for layer in self.layers[:-1]:
            x = layer.forward(x)
        return x

    def forward_propagation(self, x, y):
        xs = [x]
        for layer in self.layers[:-1]:
            x = layer.forward(x)
            xs.append(x)
        return xs, self.layers[-1].error(x, y)

    def backward_propagation(self, xs, y):
        error = self.layers[-1].backward(xs[-1], y)
        for i, layer in enumerate(reversed(self.layers[:-1])):
            if layer.is_trainable:
                args = (xs[-i - 2], error, self.learing_rate)
            else:
                args = (xs[-i - 2], error)
            error = layer.backward(*args)

    def train(self, X, Y, epochs):
        errors = []
        for _ in range(epochs):
            error = 0
            for x, y in zip(X, Y):
                x = x.reshape(x.shape[0], 1)
                y = y.reshape(y.shape[0], 1)
                xs, e = self.forward_propagation(x, y)
                self.backward_propagation(xs, y)
                error += e
            print(error / len(X), end="\r")
            errors.append(error / len(X))

        plt.plot(errors)
        plt.show()


class Layer:
    def __init__(self) -> None:
        pass

    def forward(self, input):
        pass

    def backward(self, input, error):
        pass


class TrainableLayer(Layer):
    def __init__(self) -> None:
        self.is_trainable = True
        super().__init__()


class NonTrainableLayer(Layer):
    def __init__(self) -> None:
        self.is_trainable = False
        super().__init__()


class Dense(TrainableLayer):
    def __init__(self, input_size, output_size) -> None:
        self.weights = np.random.rand(output_size, input_size)
        self.biases = np.random.rand(output_size, 1)
        self.input_size = input_size
        self.output_size = output_size
        super().__init__()

    def forward(self, x):
        assert x.shape[0] == self.input_size, "Wrong dimension of input x"
        return np.dot(self.weights, x) + self.biases

    def backward(self, x, error, learning_rate):
        dedx = np.dot(self.weights.T, error)
        self.weights -= learning_rate * np.dot(error, x.T)
        self.biases -= learning_rate * error
        return dedx


class Tanh(NonTrainableLayer):
    def __init__(self, input_size: int) -> None:
        self.input_size = input_size
        self.a = np.tanh
        self.a_p = lambda x: 1 - np.tanh(x) ** 2
        super().__init__()

    def forward(self, x):
        return self.a(x)

    def backward(self, x, error):
        return self.a_p(x) * error


class Sigmoid(NonTrainableLayer):
    def __init__(self, input_size: int) -> None:
        self.input_size = input_size
        self.a = lambda x: 1 / (1 + np.exp(-x))
        self.a_p = lambda x: self.a(x) * (1 - self.a(x))
        super().__init__()

    def forward(self, x):
        return self.a(x)

    def backward(self, x, error):
        return self.a_p(x) * error


class Softmax(NonTrainableLayer):
    def __init__(self, input_size: int) -> None:
        self.input_size = input_size
        super().__init__()

    def forward(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def backward(self, x, error):
        def softmax_prime(i, j):
            ai = np.exp(x[i]) / np.sum(np.exp(x))
            if i == j:
                return ai * (1 - ai)
            else:
                aj = np.exp(x[j]) / np.sum(np.exp(x))
                return -ai * aj

        J = []
        for i in range(self.input_size):
            J.append([])
            for j in range(self.input_size):
                J[i].append(softmax_prime(i, j))

        J = np.array(J).reshape(self.input_size, self.input_size)

        return np.dot(J, error)


class CrossEntropy(NonTrainableLayer):
    def __init__(self, input_size: int) -> None:
        self.input_size = input_size
        super().__init__()

    def error(self, x, y):
        return -np.sum(y * np.log(x))

    def backward(self, x, y):
        return -y / x


class BinaryCrossEntropy(NonTrainableLayer):
    def __init__(self, input_size: int) -> None:
        self.input_size = input_size
        super().__init__()

    def error(self, x, y):
        return -np.mean(y * np.log(x) + (1 - y) * np.log(1 - x))

    def backward(self, x, y):
        return (x - y) / (x * (1 - x) * self.input_size)


class MSE(NonTrainableLayer):
    def __init__(self, input_size: int) -> None:
        self.input_size = input_size
        super().__init__()

    def error(self, x, y):
        return 2 * np.mean(np.power(x - y, 2))

    def backward(self, x, y):
        return 2 * (x - y) / self.input_size


# Problem 1
# XOR

# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Y = np.array([[0], [1], [1], [0]])

# layers = [
#     Dense(2, 2),
#     Sigmoid(2),
#     Dense(2, 1),
#     Sigmoid(1),
#     BinaryCrossEntropy(1)
# ]

# nn = NeuNet(layers, 1)

# nn.train(X, Y, 20000)

# print(nn.forward(X[0]))
# print(nn.forward(X[1]))
# print(nn.forward(X[2]))
# print(nn.forward(X[3]))


# Binary Classifier for circular data

X = np.random.rand(400, 2)
Y = np.array([])
for x, y in X:
    if (x - 0.5) ** 2 + (y - 0.5) ** 2 < 0.2**2:
        Y = np.append(Y, [1, 0, 0])
    elif x < 0.5:
        Y = np.append(Y, [0, 1, 0])
    else:
        Y = np.append(Y, [0, 0, 1])

Y = Y.reshape(400, 3)

layers = [
    Dense(2, 3),
    Sigmoid(3),
    Dense(3, 3),
    Sigmoid(3),
    Dense(3, 3),
    Softmax(3),
    CrossEntropy(3),
]

nn = NeuNet(layers, 0.1)

nn.train(X, Y, 500)


plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=np.argmax(Y, axis=1))
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros((100, 100))
for i in range(100):
    for j in range(100):
        Z[i, j] = np.argmax(nn.forward(np.array([X[i, j], Y[i, j]])))
plt.contour(X, Y, Z, levels=[0.5, 1.5], colors="black")
plt.show()
