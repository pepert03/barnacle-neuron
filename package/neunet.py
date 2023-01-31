import json
import os
import numpy as np
import matplotlib.pyplot as plt


class NeuNet:
    def __init__(self, layers: list=[], learing_rate: float=0.1) -> None:
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
        for epoch in range(epochs):
            error = 0
            for x, y in zip(X, Y):
                x = x.reshape(x.shape[0], 1)
                y = y.reshape(y.shape[0], 1)
                xs, e = self.forward_propagation(x, y)
                self.backward_propagation(xs, y)
                error += e
            print(epoch, ":", error / len(X), end="\r")
            errors.append(error / len(X))
        print(epoch, ":", error / len(X))
        return errors

    def test(self, X, Y):
        correct = 0
        for x, y in zip(X, Y):
            x = x.reshape(x.shape[0], 1)
            y = y.reshape(y.shape[0], 1)
            y_true = np.argmax(y)
            y_pred = np.argmax(self.forward(x))
            if y_true == y_pred:
                correct += 1
        return correct / len(X)

    def save(self, model_name, folder="models"):
        if not os.path.exists(folder):
            os.makedirs(folder)
        folder = os.path.join(folder, model_name)
        if not os.path.exists(folder):
            os.makedirs(folder)
        for i, layer in enumerate(self.layers):
            if layer.is_trainable:
                np.save(os.path.join(folder, f"weights{i}.npy"), layer.weights)
                np.save(os.path.join(folder, f"biases{i}.npy"), layer.biases)
        info = json.dumps({
            "layers": [layer.save_dict() for layer in self.layers],
            "learing_rate": self.learing_rate
        }, indent=4)
        with open(os.path.join(folder, "info.json"), "w") as f:
            f.write(info)

    def load(self, model_name, folder="models"):
        folder = os.path.join(folder, model_name)
        with open(os.path.join(folder, "info.json"), "r") as f:
            info = json.load(f)
        self.layers = []
        for i, layer in enumerate(info["layers"]):
            layer_class = globals()[layer["name"]]
            layer = layer_class(*layer["args"])
            if layer.is_trainable:
                layer.weights = np.load(os.path.join(folder, f"weights{i}.npy"))
                layer.biases = np.load(os.path.join(folder, f"biases{i}.npy"))
            self.layers.append(layer)
        self.learing_rate = info["learing_rate"]


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
    
    def save_dict(self):
        return {"name": self.__class__.__name__, "args": [self.input_size, self.output_size]}


class NonTrainableLayer(Layer):
    def __init__(self) -> None:
        self.is_trainable = False
        super().__init__()

    def save_dict(self):
        return {"name": self.__class__.__name__, "args": [self.input_size]}


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


class Normalization(NonTrainableLayer):
    def __init__(self, input_size: int) -> None:
        self.input_size = input_size
        self.a = lambda x: x / np.sum(x)
        self.a_p = lambda x: (1 / np.sum(x)) - (x / np.sum(x) ** 2)
        super().__init__()

    def forward(self, x):
        return self.a(x)

    def backward(self, x, error):
        return self.a_p(x) * error