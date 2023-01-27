import sys
import os
import numpy as np
import random

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from package.neunet import *


def train_test_split(X, Y, test_size=0.2):
    X_train, X_test, Y_train, Y_test = [], [], [], []
    for i in range(len(X)):
        if random.random() < test_size:
            X_test.append(X[i])
            Y_test.append(Y[i])
        else:
            X_train.append(X[i])
            Y_train.append(Y[i])
    return np.array(X_train), np.array(X_test), np.array(Y_train), np.array(Y_test)


X = np.load("data/X.npy")
Y = np.load("data/Y.npy")

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


layers = [
    Dense(784, 16),
    Tanh(16),
    Dense(16, 10),
    Tanh(10),
    Dense(10, 10),
    Softmax(10),
    CrossEntropy(10),
]

nn = NeuNet(layers, 0.1)

# Train
errors = nn.train(X_train, Y_train, 100)

# Test
acc = nn.test(X_test, Y_test)
print(f"Accuracy: {acc}")

# Plot
plt.plot(errors)
plt.show()
