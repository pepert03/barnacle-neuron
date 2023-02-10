import json
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from package import utils


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

    def save_dict(self):
        return {"name": self.__class__.__name__, "args": []}

    @classmethod
    def load_dict(cls, d):
        return cls(*d["args"])


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

    @classmethod
    def load_dict(cls, d):
        return cls(*d["args"])


class NonTrainableLayer(Layer):
    def __init__(self) -> None:
        self.is_trainable = False
        super().__init__()

    def save_dict(self):
        return {"name": self.__class__.__name__, "args": [self.input_size]}

    @classmethod
    def load_dict(cls, d):
        return cls(*d["args"])


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

    def save_dict(self):
        """Save weights and biases to file and return a dictionary with the paths to the files"""
        weights_path = os.path.join(
            self.save_dir, f"{__class__.__name__}{self.layer_id}_weights.npy"
        )
        biases_path = os.path.join(
            self.save_dir, f"{__class__.__name__}{self.layer_id}_biases.npy"
        )
        np.save(weights_path, self.weights)
        np.save(biases_path, self.biases)
        return (
            super()
            .save_dict()
            .update(
                {
                    "weights_path": weights_path,
                    "biases_path": biases_path,
                }
            )
        )

    @classmethod
    def load_dict(cls, d):
        """Load weights and biases from files and return a new Dense layer"""
        weights = np.load(d["weights_path"])
        biases = np.load(d["biases_path"])
        lay = cls(*d["args"])
        lay.weights = weights
        lay.biases = biases
        return lay


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
        # Compiled
        self.compiled = False

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

        # Compiled
        self.compiled = True

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

        assert self.compiled, "Network not compiled"

        # Train the network
        errors = []
        for epoch in range(epochs):
            error = 0
            # Shuffle the data to avoid overfitting
            idx = np.random.permutation(len(X))
            # Store Y_pred for each training example
            Y_pred = np.zeros(Y.shape)
            # For each training example
            for i, (x, y) in enumerate(zip(X[idx], Y[idx])):
                x = x.reshape(x.shape[0], 1)
                y = y.reshape(y.shape[0], 1)
                y_pred, e = self.forward_propagation(x, y)
                self.backward_propagation(y)
                Y_pred[i] = y_pred.reshape(y_pred.shape[0])
                error += e
            error = error / len(X)
            errors.append(error)
            metrics = self.calculate_metrics(Y, Y_pred)
            if verbose:
                print(epoch, ":", error, metrics, end="\r")
        if verbose:
            print(epoch, ":", error, metrics)
        return errors

    def calculate_metrics(self, Y, Y_pred):
        """
        Calculates the metrics for the given Y and Y_pred
        """
        map_metric = {
            "accuracy": utils.accuracy,
            "precision": utils.precision,
            "recall": utils.recall,
            "f1_score": utils.f1_score,
        }
        metrics = {}
        for metric_name in self.metrics:
            metric = map_metric[metric_name]
            metrics[metric_name] = metric(Y, Y_pred)
        return metrics

    def test(self, X_test, Y_test):
        """
        Tests the network on the given data X_test and Y_test using the
        metrics specified in the compile method.
        """
        Y_pred = np.zeros(Y_test.shape)
        for i, (x, y) in enumerate(zip(X_test, Y_test)):
            x = x.reshape(x.shape[0], 1)
            y = y.reshape(y.shape[0], 1)
            y_pred, _ = self.forward_propagation(x, y)
            Y_pred[i, :] = y_pred
        return self.calculate_metrics(Y_test, Y_pred)

    def save(self, model_name, folder="models"):
        """
        Saves the model to the given folder with the given name.
        """

        # Create folders if they don't exist
        if not os.path.exists(folder):
            os.makedirs(folder)

        model_path = os.path.join(folder, model_name)

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # Save each layer. If the layer is trainable, save its weights.
        info = json.dumps(
            {
                "layers": [layer.save_dict() for layer in self.layers],
                "learing_rate": self.learing_rate,
                "metrics": self.metrics,
                "compiled": self.compiled,
            },
            indent=4,
        )

        with open(os.path.join(model_path, "info.json"), "w") as f:
            f.write(info)

    @classmethod
    def load(cls, model_name, folder="models"):
        folder = os.path.join(folder, model_name)
        with open(os.path.join(folder, "info.json"), "r") as f:
            info = json.load(f)
        layers = []
        for layer_info in info["layers"]:
            class_name = layer_info["name"]
            layer_class = globals()[class_name]
            layer = layer_class.load_dict(layer_info)
            layers.append(layer)
        model = cls(layers)
        model.compiled = info["compiled"]
        model.compile(info["learing_rate"], info["metrics"])
        return model


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
