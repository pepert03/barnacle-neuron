import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from package.neunet import *

# Build Dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

# Network Arquitecture


n, c = 0, 0
for _ in range(250):
    layers = [
        Dense(2, 3),
        Tanh(3),
        Dense(3, 1),
        Tanh(1),
        MSE(1),
    ]
    nn = NeuNet(layers)

    # Compile
    nn.compile(learning_rate=0.1, metrics=["accuracy","recall","precision"])

    # Train
    errors = nn.fit(X, Y, epochs=1000)

    # Check if the network is working
    all_correct = True
    for i in range(4):
        y_ = nn.forward(X[i])
        if y_ > 0.5:
            if Y[i] < 0.5:
                all_correct = False
        else:
            if Y[i] > 0.5:
                all_correct = False
        # print(f"{list(X[i])} -> {float(y_):.3f} =? {list(Y[i])}")
    if all_correct:
        c += 1
    n += 1
    # print(c, n)
    print(f"Converging {c}/{n}: ", f"{100*c/n:.3f} %", end="\r")
    # input()
print(f"Converging {c}/{n}: ", f"{100*c/n:.3f} %")
