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


class NonTrainableLayer(Layer):
    def __init__(self) -> None:
        self.is_trainable = False
        super().__init__()


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

    def backward(self, derror):
        dedx = np.dot(self.weights.T, derror)
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


# class Softmax(NonTrainableLayer):
#     def __init__(self, input_size: int) -> None:
#         self.input_size = input_size
#         self.last_x = None
#         self.last_a = None
#         super().__init__()

#     def forward(self, x):
#         self.last_x = x
#         self.last_a = np.exp(x) / np.sum(np.exp(x))
#         return self.last_a

#     def backward(self, derror):
#         ai = self.last_a
#         J = np.diag(ai) - np.dot(ai, ai.T)
#         return np.dot(J.T, derror)


# class CrossEntropy(NonTrainableLayer):
#     def __init__(self, input_size: int) -> None:
#         self.input_size = input_size
#         self.last_x = None
#         super().__init__()

#     def error(self, x, y):
#         self.last_x = x
#         return -np.sum(y * np.log(x))

#     def backward(self, y):
#         return -y / self.last_x


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

        return np.dot(J, error)


class CrossEntropy(NonTrainableLayer):
    def __init__(self, input_size: int) -> None:
        self.input_size = input_size
        self.last_x = None
        super().__init__()

    def error(self, x, y):
        self.last_x = x
        return -np.sum(y * np.log(x))

    def backward(self, y):
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
    def __init__(self, layers: list, learing_rate, verbose=True) -> None:
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

    def untrain(self, y):
        """Trains the input to maximize the activation"""

    def train(self, X, Y, epochs):
        errors = []
        for epoch in range(epochs):
            error = 0
            for x, y in zip(X, Y):
                x = x.reshape(x.shape[0], 1)
                y = y.reshape(y.shape[0], 1)
                # print("Forward propagation...")
                e = self.forward_propagation(x, y)
                # print("Backward propagation...")
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
