import json
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.signal import convolve

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from package import utils


class Layer:
    def __init__(self):
        pass

    def wrap_forward(self, func):
        """Decorator to wrap forward methods. Debugging purposes"""

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

    def wrap_backward(self, func):
        """Decorator to wrap backward methods. Debugging purposes"""

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

    def save_dict(self, **_):
        """Saves itself and returns a dictionary with the information needed to recreate it"""
        return {"class_name": self.__class__.__name__, "init_args": []}

    @classmethod
    def load_dict(cls, d):
        """Loads itself from the dictionary returned by save_dict"""
        return cls(*d["init_args"])


class TrainableLayer(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.is_trainable = True
        self.is_loss = False


class ActivationLayer(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.is_trainable = False
        self.is_loss = False
        self.output_size = self.input_size

    def save_dict(self, **_):
        """Saves itself and returns a dictionary with the information needed to recreate it"""
        return {"class_name": self.__class__.__name__, "init_args": [self.input_size]}

    @classmethod
    def load_dict(cls, d):
        """Loads itself from the dictionary returned by save_dict"""
        return cls(*d["init_args"])


class LossLayer(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.is_trainable = False
        self.is_loss = True

    def save_dict(self, **_):
        """Saves itself and returns a dictionary with the information needed to recreate it"""
        return {"class_name": self.__class__.__name__, "init_args": [self.input_size]}

    @classmethod
    def load_dict(cls, d):
        """Loads itself from the dictionary returned by save_dict"""
        return cls(*d["init_args"])


class Dense(TrainableLayer):
    def __init__(self, input_size, output_size) -> None:
        # Initialize weights and biases between -1 and 1
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

    def save_dict(self, **kwargs):
        """Save weights and biases to file and return a dictionary with the paths to the files
        and all the other information needed to recreate the layer
        """
        # Get the model path from the kwargs
        model_path = kwargs["model_path"]
        assert model_path is not None, "model_path must be specified"
        assert os.path.isdir(model_path), "model_path must be a valid directory"

        # Save weights and biases to files
        weights_path = os.path.join(
            model_path, f"{__class__.__name__}{self.id}_weights.npy"
        )
        biases_path = os.path.join(
            model_path, f"{__class__.__name__}{self.id}_biases.npy"
        )
        np.save(weights_path, self.weights)
        np.save(biases_path, self.biases)

        return {
            "class_name": self.__class__.__name__,
            "init_args": [self.input_size, self.output_size],
            "weights_path": weights_path,
            "biases_path": biases_path,
        }

    @classmethod
    def load_dict(cls, d):
        """Loads itself from the dictionary returned by save_dict. Also loads the weights and biases."""
        layer = cls(*d["init_args"])
        weights = np.load(d["weights_path"])
        biases = np.load(d["biases_path"])
        layer.weights = weights
        layer.biases = biases
        return layer


class Convolutional(TrainableLayer):
    def __init__(self, input_size, output_size, kernel_size) -> None:
        # Initialize weights and biases between -1 and 1
        self.weights = (
            2 * np.random.rand(output_size, input_size[2], kernel_size, kernel_size) - 1
        )
        self.biases = 2 * np.random.rand(output_size, 1) - 1
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.last_x = None
        super().__init__()


class Tanh(ActivationLayer):
    def __init__(self, input_size: int) -> None:
        self.input_size = input_size
        self.last_a = None
        super().__init__()

    def forward(self, x):
        self.last_a = np.tanh(x)
        return self.last_a

    def backward(self, derror):
        return (1 - self.last_a**2) * derror


class Sigmoid(ActivationLayer):
    def __init__(self, input_size: int) -> None:
        self.input_size = input_size
        self.last_a = None
        super().__init__()

    def forward(self, x):
        self.last_a = 1 / (1 + np.exp(-x))
        return self.last_a

    def backward(self, derror):
        return self.last_a * (1 - self.last_a) * derror


class Softmax(ActivationLayer):
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


class ReLU(ActivationLayer):
    def __init__(self, input_size: int, a=0) -> None:
        self.input_size = input_size
        self.a = a
        self.last_da = None
        self.ones = np.ones((self.input_size, 1))
        super().__init__()

    def forward(self, x):
        self.last_da = self.ones.copy()
        self.last_da[x < 0] = self.a
        return self.last_da * x

    def backward(self, error):
        return self.last_da * error


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
    def __init__(self, layers: list[Layer] = [], loss_layer: Layer = None) -> None:
        # Check layers
        assert len(layers) > (
            loss_layer is None
        ), "Network must have at least one layer"
        assert layers[-1].is_loss | isinstance(
            loss_layer, LossLayer
        ), "You must specify a loss layer"

        # Network layers
        if loss_layer is not None:
            self.layers = layers
            self.loss_layer = loss_layer
        else:
            self.layers, self.loss_layer = layers[:-1], layers[-1]
            assert len(self.layers) > 0, "Network must have at least one layer"

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
            layer.id = i
            if layer.is_trainable:
                layer.learning_rate = learning_rate

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

    def fit(self, X, Y, epochs=1, verbose=True):
        """
        Trains the network on the given data X and Y for the given number of
        epochs.

        Returns a list of errors for each epoch.
        """

        assert self.compiled, "Network not compiled"

        if verbose:
            os.system("")
            ANSI_HIDE_CURSOR = "\x1b[?25l"
            print("Train:", ANSI_HIDE_CURSOR)

        # Train the network
        errors = []

        for epoch in range(epochs):
            # Reset error
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
                Y_pred[idx[i]] = y_pred.reshape(y_pred.shape[0])
                error += e

            # Calculate the error and metrics
            error = error / len(X)
            errors.append(error)
            metrics = self.calculate_metrics(Y, Y_pred)

            if verbose:
                print(
                    f"{' '*(len(str(epochs))-len(str(epoch + 1)))+str(epoch + 1)}/{epochs}",
                    f"{'█' * int((epoch+1) / epochs * 30)}{'░' * (30 - int((epoch+1) / epochs * 30))}",
                    f"- loss: {error:.4f}",
                    " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]),
                    end="\r" if epoch < epochs - 1 else "\n",
                )

        if verbose:
            ANSI_SHOW_CURSOR = "\x1b[?25h"
            print(ANSI_SHOW_CURSOR, end="")

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
            if metric_name not in map_metric:
                print(f"Warning: metric '{metric_name}' not found")
            metric = map_metric[metric_name]
            metrics[metric_name] = metric(Y, Y_pred)
        return metrics

    def evaluate(self, X_test, Y_test, verbose=True):
        """
        Evaluates the network on the given data X_test and Y_test using the
        metrics specified in the compile method.
        Returns the loss of the network and the metrics.
        """
        if verbose:
            os.system("")
            ANSI_HIDE_CURSOR = "\x1b[?25l"
            print("Test:", ANSI_HIDE_CURSOR)

        # Evaluate the network
        error = 0
        Y_pred = np.zeros(Y_test.shape)
        N = len(X_test)

        # For each training example
        for i, (x, y) in enumerate(zip(X_test, Y_test)):

            x = x.reshape(x.shape[0], 1)
            y = y.reshape(y.shape[0], 1)
            y_pred, e = self.forward_propagation(x, y)
            Y_pred[i] = y_pred.reshape(y_pred.shape[0])

            error += e

            if verbose:
                print(
                    f"{' '*(len(str(N))-len(str(i + 1)))+str(i + 1)}/{N}",
                    f"{'█' * int((i+1) / N * 30)}{'░' * (30 - int((i+1) / N * 30))}",
                    f"- loss: {error/(i+1):.4f}",
                    end="\r",
                )

        # Calculate metrics
        error = error / len(X_test)
        metrics = self.calculate_metrics(Y_test, Y_pred)

        if verbose:
            ANSI_SHOW_CURSOR = "\x1b[?25h"
            print(
                f"{' '*(len(str(N))-len(str(N)))+str(N)}/{N}",
                f"{'█' * int((N) / N * 30)}{'░' * (30 - int((N) / N * 30))}",
                f"- loss: {error:.4f}",
                " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]),
                ANSI_SHOW_CURSOR,
            )

        return error, metrics

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

    def save(self, model_path: str):
        """
        Saves the model to the given folder with the given name.
        Structure of the folder:
            - model.json
            - Weights0.npy
            - Weights1.npy
            - ...
        """

        assert self.compiled, "Network not compiled"

        # Create folder if they don't exist
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # Model info
        layers_dicts = [layer.save_dict(model_path=model_path) for layer in self.layers]
        model_dict = json.dumps(
            {
                "layers": layers_dicts,
                "loss_layer": self.loss_layer.save_dict(),
                "learing_rate": self.learing_rate,
                "metrics": self.metrics,
            },
            indent=4,
        )

        # Save model
        with open(os.path.join(model_path, "model.json"), "w") as f:
            f.write(model_dict)

    @classmethod
    def load(cls, model_path: str):
        """
        Loads the model with the given name from the given folder.
        """

        # Load info dict
        with open(os.path.join(model_path, "model.json"), "r") as f:
            info = json.load(f)

        # Create layers
        layers = []
        for layer_dict in info["layers"]:
            class_name = layer_dict["class_name"]
            layer_class = globals()[class_name]
            layer = layer_class.load_dict(layer_dict)
            layers.append(layer)

        # Create loss layer
        loss_layer_name = info["loss_layer"]["class_name"]
        loss_layer = globals()[loss_layer_name].load_dict(info["loss_layer"])

        # Create model
        model = cls(layers, loss_layer)
        model.compile(info["learing_rate"], info["metrics"])

        return model


def debug_layer_init(self) -> None:
    if not hasattr(self, "forward"):
        self.forward = self.error
        self.error = self.wrap_forward(self.error)
    self.forward = self.wrap_forward(self.forward)
    self.backward = self.wrap_backward(self.backward)


def normal_layer_init(self) -> None:
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
