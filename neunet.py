import numpy as np
import random as r
import pandas as pd

class NeuNet:
    def __init__(self) -> None:
        pass

class Layer:
    def __init__(self) -> None:
        pass

    def forward(self, input):
        pass

    def backward(self, input, error):
        pass

class Dense(Layer):
    def __init__(self,input_size,output_size) -> None:
        self.weights = np.random.rand(input_size,output_size)
        self.biases = np.random.rand(output_size,1)
        self.input_size=input_size
        self.output_size=output_size
        super().__init__()
    
    def forward(self, x):
        assert x.size[0]==self.input_size,"Wrong dimension of input x"
        return np.dot(self.weights,x)+self.biases
    
    def backward(self, input, error, learning_rate):
        self.weights -= learning_rate*np.dot(error,input)
        self.biases -= learning_rate*error
        return np.dot(self.weights.T,error)


class Activation(Layer):
    def __init__(self):
        pass

    def forward(self, input):
        pass

    def backward(self, input, error):
        pass
