import sys
import os
import pandas as pd
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

df = pd.read_csv("data/data.csv")
X = df['data'].to_numpy()
Y = df['label'].to_numpy()

for i in range(len(X)):
    X[i] = np.array(eval(X[i]))
    Y[i] = np.array(eval(Y[i]))

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

errors = nn.train(X, Y, 100)

# Test
correct = 0
for i in range(len(X_test)):
    y_true = np.argmax(Y_test[i])
    y_pred = np.argmax(nn.forward(X_test[i]))
    print(f"True: {y_true}, Predicted: {y_pred}")
    if y_true == y_pred:
        correct += 1

print(f"Accuracy: {correct / len(X_test)}")

