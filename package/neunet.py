import json
import os
import numpy as np
import matplotlib.pyplot as plt


class Layer:
    def __init__(self):
        pass

    # Decorator to wrap forward method. Debugging purposes
    def wrap_forward(self, func):
        def wrapper(*args, **kwargs):
            print(f"\n{self.__class__.__name__} Layer doing Forward Propagation...\n")
            print(f"\tArgs:")
            for _, arg in enumerate(args):
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

    # Decorator to wrap backward methods. Debugging purposes
    def wrap_backward(self, func):
        def wrapper(*args, **kwargs):
            print(f"\n{self.__class__.__name__} Layer doing Backward Propagation...\n")
            print(f"\tArgs:")
            for _, arg in enumerate(args):
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
        self.is_loss_layer = False
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


class LossLayer(NonTrainableLayer):
    def __init__(self) -> None:
        self.is_loss_layer = True
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

    def backward(self, derror, update=True):
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
        self.last_a = None
        super().__init__()

    def forward(self, x):
        self.last_a = np.exp(x) / np.sum(np.exp(x))
        return self.last_a

    def backward(self, error):
        ai = self.last_a
        J = np.diagflat(ai) - np.outer(ai, ai)
        return np.dot(J, error)


class CrossEntropy(LossLayer):
    def __init__(self, input_size: int) -> None:
        self.input_size = input_size
        self.last_x = None
        super().__init__()

    def error(self, x, y):
        self.last_x = x
        return -np.sum(y * np.log(x))

    def backward(self, y):
        return -y / self.last_x


class BinaryCrossEntropy(LossLayer):
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


class MSE(LossLayer):
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
    def __init__(self, layers: list[Layer] = []) -> None:
        # Network layers
        self.layers = layers

    def compile(self, learning_rate: float = 0.1, metrics: list = []):
        """
        Initializes the network by setting the learning rate and metrics to use
        when training and testing the network.
        """
        # Set learning rate
        self.learing_rate = learning_rate
        for i, layer in enumerate(self.layers):
            layer.index = i
            if layer.is_trainable:
                layer.learning_rate = learning_rate

        # Get loss layer
        self.layers, self.loss_layer = self.layers[:-1], self.layers[-1]

        # Metrics
        self.metrics = metrics

    def predict(self, x):
        """
        Makes a forward pass through the network given an input vector x and
        returns the output vector of the last layer.
        """
        x = x.reshape(x.shape[0], 1)
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward_propagation(self, x, y):
        """Makes a forward pass through the network and returns the error"""
        for layer in self.layers:
            x = layer.forward(x)
        return x, self.loss_layer.error(x, y)

    def backward_propagation(self, y):
        """
        Makes a backward pass through the network and returns the imputed error.
        Every trainable layer updates its weights and biases.
        """
        derror = self.loss_layer.backward(y)
        for layer in reversed(self.layers):
            derror = layer.backward(derror)
        return derror

    def untrain(self, y, learning_rate=0.1, epochs=1000, error_plot=True):
        """Finds the input that maximizes the output of the last layer."""

        # Random input
        input_size = self.layers[0].input_size
        inp = 2 * np.random.rand(input_size, 1) - 1

        # Untrain
        errors = []
        for _ in range(epochs):
            e = self.forward_propagation(inp, y)
            derror = self.loss_layer.backward(y)
            for layer in reversed(self.layers):
                if layer.is_trainable:
                    derror = layer.backward(derror, update=False)
                else:
                    derror = layer.backward(derror)
            inp -= learning_rate * derror
            errors.append(e)

        # Error plot
        if error_plot:
            plt.plot(errors)
            plt.show()

        return inp

    def fit(self, X, Y, epochs=1, verbose=False):
        """
        Trains the network on the given data X and Y for the given number of
        epochs.

        Returns a list of errors for each epoch.
        """
        # Train the network
        errors = []
        for epoch in range(epochs):
            error = 0
            # For each training example
            for x, y in zip(X, Y):
                x = x.reshape(x.shape[0], 1)
                y = y.reshape(y.shape[0], 1)
                y_pred, e = self.forward_propagation(x, y)
                self.backward_propagation(y)
                error += e
            error = error / len(X)
            errors.append(error)
            if verbose:
                print(epoch, ":", error, end="\r")
        if verbose:
            print(epoch, ":", error)
        return errors

    def test(self, X, Y):
        """
        TODO: Implement different metrics
        -
        -
        -
        -
        -
        -
        -
        """
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
        """
        Saves the model to the given folder with the given name.
        """
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

    @classmethod
    def load(cls, model_name, folder="models"):
        folder = os.path.join(folder, model_name)
        with open(os.path.join(folder, "info.json"), "r") as f:
            info = json.load(f)
        layers = []
        for i, layer in enumerate(info["layers"]):
            layer_class = globals()[layer["name"]]
            layer = layer_class(*layer["args"])
            if layer.is_trainable:
                layer.weights = np.load(os.path.join(folder, f"weights{i}.npy"))
                layer.biases = np.load(os.path.join(folder, f"biases{i}.npy"))
            layers.append(layer)
        learing_rate = info["learing_rate"]
        return cls(layers, learing_rate)


def debug_layer_init(self) -> None:
    if not hasattr(self, "forward"):
        self.forward = self.error
        self.error = self.wrap_forward(self.error)
    self.forward = self.wrap_forward(self.forward)
    self.backward = self.wrap_backward(self.backward)


def normal_layer_init(_) -> None:
    pass


def activate_debug_mode():
    """
    Sets the debug mode for all layers.
    This will print the shapes of the inputs and outputs of each layer, in forward and backward pass.
    """
    Layer.__init__ = debug_layer_init


def deactivate_debug_mode():
    """
    Sets the normal mode for all layers.
    """
    Layer.__init__ = normal_layer_init
