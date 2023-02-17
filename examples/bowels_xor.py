import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from package.neunet import *

# numpy warnings to errors
np.seterr(all="raise")

# Create network
nn = NeuNet.load("models/xor")

# Create figure
fig = plt.figure("NeuralNetwork Bowels", figsize=(12, 5))

# 2D Plot
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

ax = plt.subplot(1, 1, 1)

ax.set_xlabel("x1")
ax.set_ylabel("x2")

# Background color based on Z
ax.scatter(X[:, 0], X[:, 1], c=Y > 0.5, cmap="bwr")
xs = np.linspace(-5, 5, 100)
ys = np.linspace(-5, 5, 100)
Xg, Yg = np.meshgrid(xs, ys)
Z = np.zeros((100, 100))
for i in range(100):
    for j in range(100):
        Z[i, j] = nn.predict(np.array([Xg[i, j], Yg[i, j]]))

ax.imshow(Z, extent=[-5, 5, -5, 5], origin="lower", alpha=0.2)

for i in range(2):
    # Train the input to maximize the output
    y = i
    inp = nn.untrain(y, 0.05, 5000, error_plot=False)
    print(inp)
    ax.plot(*inp, "*", label=f"Output {i}")

plt.legend()
plt.show()
