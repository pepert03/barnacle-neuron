import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from package.neunet import *

# numpy warnings to errors
np.seterr(all="raise")

# Create network
nn = NeuNet.load("models/mnist")

fig = plt.figure("NeuralNetwork Bowels", figsize=(12, 5))

for i in range(10):
    y = np.zeros((10, 1))
    y[i] = 1
    inp = nn.untrain(y, 0.05, 5000, error_plot=False)
    plt.subplot(2, 5, i + 1)
    plt.imshow(inp.reshape(28, 28), cmap="gray")
    plt.title(f"Number: {i}")
    plt.axis("off")
    # Feed input to the network
    print("Feeding:", i)
    print(nn.predict(inp))

plt.show()
