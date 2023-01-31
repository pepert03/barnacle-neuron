import json
import os
import numpy as np
import matplotlib.pyplot as plt


class Layer:
    def __init__(self) -> None:
        # if not hasattr(self, "forward"):
        #     self.forward = self.error
        #     self.error = self.wrap_forward(self.error)
        # self.forward = self.wrap_forward(self.forward)
        # self.backward = self.wrap_backward(self.backward)
        pass

    # Add a decorator to wrap forward and backward methods
    def wrap_forward(self, func):
        def wrapper(*args, **kwargs):
            print(f"\n{self.__class__.__name__} Layer doing Forward Propagation...\n")
            print(f"\tArgs:")
            for i, arg in enumerate(args):
                print("-" * 20)
                print(arg)
            print("-" * 20)
            out = func(*args, **kwargs)
            print("\tOutput:")
            print("-" * 20)
            print(out)
            print("-" * 20)
            input()
            return out

        return wrapper

    def wrap_backward(self, func):
        def wrapper(*args, **kwargs):
            print(f"\n{self.__class__.__name__} Layer doing Backward Propagation...\n")
            print(f"\tArgs:")
            for i, arg in enumerate(args):
                print("-" * 20)
                print(arg)
            print("-" * 20)
            out = func(*args, **kwargs)
            print("\tOutput:")
            print("-" * 20)
            print(out)
            print("-" * 20)
            input()
            return out

        return wrapper


class TrainableLayer(Layer):
    def __init__(self) -> None:
        self.is_trainable = True
        super().__init__()

    def save_dict(self):
        return {
            "name": self.__class__.__name__,
            "args": [self.input_size, self.output_size],
        }


class NonTrainableLayer(Layer):
    def __init__(self) -> None:
        self.is_trainable = False
        super().__init__()

    def save_dict(self):
        return {"name": self.__class__.__name__, "args": [self.input_size]}


class Dense(TrainableLayer):
    def __init__(self, input_size, output_size) -> None:
        self.weights = 2 * np.random.rand(output_size, input_size) - 1
        self.biases = 2 * np.random.rand(output_size, 1) - 1
        self.input_size = input_size
        self.output_size = output_size
        self.last_x = None
        super().__init__()

    def forward(self, x):
        assert x.shape[0] == self.input_size, "Wrong dimension of input x"
        self.last_x = x
        return np.dot(self.weights, x) + self.biases

    def backward(self, derror, update=True):
        print(self.weights.shape, derror.shape, self.last_x.shape)
        dedx = np.dot(self.weights.T, derror)
        if update:
            self.weights -= self.learning_rate * np.dot(derror, self.last_x.T)
            self.biases -= self.learning_rate * derror
        return dedx


class Tanh(NonTrainableLayer):
    def __init__(self, input_size: int) -> None:
        self.input_size = input_size
        self.last_a = None
        super().__init__()

    def forward(self, x):
        self.last_a = np.tanh(x)
        return self.last_a

    def backward(self, derror):
        return (1 - self.last_a**2) * derror


class Sigmoid(NonTrainableLayer):
    def __init__(self, input_size: int) -> None:
        self.input_size = input_size
        self.last_a = None
        super().__init__()

    def forward(self, x):
        self.last_a = 1 / (1 + np.exp(-x))
        return self.last_a

    def backward(self, derror):
        return self.last_a * (1 - self.last_a) * derror


class Softmax(NonTrainableLayer):
    def __init__(self, input_size: int) -> None:
        self.input_size = input_size
        self.last_x = None
        super().__init__()

    def forward(self, x):
        self.last_x = x
        return np.exp(x) / np.sum(np.exp(x))

    def backward(self, error):
        x = self.last_x
        norm = np.sum(np.exp(x))

        def softmax_prime(i, j):
            ai = np.exp(x[i]) / norm
            if i == j:
                return ai * (1 - ai)
            else:
                aj = np.exp(x[j]) / norm
                return -ai * aj

        J = []
        for i in range(self.input_size):
            J.append([])
            for j in range(self.input_size):
                J[i].append(softmax_prime(i, j))

        J = np.array(J).reshape(self.input_size, self.input_size)
        print("J:", J.shape)
        print("error:", error.shape)
        return np.dot(J.T, error)


class CrossEntropy(NonTrainableLayer):
    def __init__(self, input_size: int) -> None:
        self.input_size = input_size
        self.last_x = None
        super().__init__()

    def error(self, x, y):
        self.last_x = x
        return -np.sum(y * np.log(x))

    def backward(self, y):
        print("Error")
        print(y.shape)
        print(self.last_x.shape)
        print((-y / self.last_x).shape)
        return -y / self.last_x


class BinaryCrossEntropy(NonTrainableLayer):
    def __init__(self, input_size: int) -> None:
        self.input_size = input_size
        self.last_x = None
        super().__init__()

    def error(self, x, y):
        self.last_x = x
        return -np.mean(y * np.log(x) + (1 - y) * np.log(1 - x))

    def backward(self, y):
        x = self.last_x
        return (x - y) / (x * (1 - x) * self.input_size)


class MSE(NonTrainableLayer):
    def __init__(self, input_size: int) -> None:
        self.input_size = input_size
        self.last_x = None
        super().__init__()

    def error(self, x, y):
        self.last_x = x
        return 2 * np.mean(np.power(x - y, 2))

    def backward(self, y):
        x = self.last_x
        return 2 * (x - y) / self.input_size


class NeuNet:
    def __init__(self, layers: list = [], learing_rate=0.1, verbose=True) -> None:
        self.layers = layers
        self.learing_rate = learing_rate
        for layer in self.layers:
            if layer.is_trainable:
                layer.learning_rate = learing_rate
        self.verbose = verbose

    def forward(self, x):
        x = x.reshape(x.shape[0], 1)
        for layer in self.layers[:-1]:
            x = layer.forward(x)
        return x

    def forward_propagation(self, x, y):
        for layer in self.layers[:-1]:
            x = layer.forward(x)
        return self.layers[-1].error(x, y)

    def backward_propagation(self, y):
        derror = self.layers[-1].backward(y)
        for layer in reversed(self.layers[:-1]):
            derror = layer.backward(derror)

    def untrain(self, y, alpha=0.1, epochs=1000):
        print("WTF:", y.shape)
        input_size = self.layers[0].input_size
        inp = 2 * np.random.rand(input_size, 1) - 1
        for _ in range(epochs):
            self.forward_propagation(inp, y)
            derror = self.layers[-1].backward(y)
            for layer in reversed(self.layers):
                print(layer.__class__.__name__)
                if layer.is_trainable:
                    derror = layer.backward(derror, update=False)
                else:
                    derror = layer.backward(derror)
            print(inp.shape, derror.shape)
            inp -= alpha * derror

        # Display the image
        print("\n" * 2)
        print(inp)
        plt.imshow(inp.reshape(28, 28), cmap="gray")
        plt.show()

        return inp

    def train(self, X, Y, epochs):
        errors = []
        for epoch in range(epochs):
            error = 0
            for x, y in zip(X, Y):
                x = x.reshape(x.shape[0], 1)
                y = y.reshape(y.shape[0], 1)
                e = self.forward_propagation(x, y)
                self.backward_propagation(y)
                error += e
            if self.verbose:
                print(epoch, ":", error / len(X), end="\r")
            errors.append(error / len(X))
        if self.verbose:
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
        info = json.dumps(
            {
                "layers": [layer.save_dict() for layer in self.layers],
                "learing_rate": self.learing_rate,
            },
            indent=4,
        )
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
