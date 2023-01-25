import numpy as np
import random as r
import pandas as pd


class NeuNet:
    def __init__(self) -> None:
        pass


class Layer:
    def __init__(self) -> None:
        self.weights = np.array([])
        self.biases = np.array([])

    def forward(self, input):
        pass

    def backward(self, input, error):
        pass
