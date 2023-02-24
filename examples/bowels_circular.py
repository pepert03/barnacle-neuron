import sys
import os
import scipy

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from package.neunet import *

# numpy warnings to errors
np.seterr(all="raise")


def untrain_fcl(
    error_layer: type = MSE,
    layer_index: int = -1,
    learning_rate: float = 0.1,
    epochs: int = 1000,
):
    nn = NeuNet.load("models/circular3class")
    trainer = NeuNet.load("models/circular3class")

    assert layer_index < len(nn.layers)

    print(nn.layers, len(nn.layers), layer_index)

    layer_index = (layer_index + len(nn.layers)) % len(nn.layers)
    print(layer_index)

    trainer.layers = trainer.layers[: layer_index + 1]
    print(trainer.layers)

    error_layer = error_layer(len(trainer.layers[-1].output))


error_layer = MSE

untrain_fcl(
    output_index=0, layer_index=1, learning_rate=0.1, epochs=1000, error_plot=True
)
