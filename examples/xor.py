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
layers = [
    Dense(2, 3),
    Tanh(3),
    Dense(3, 1),
    Tanh(1),
    MSE(1),
]

nn = NeuNet(layers, 0.1)

# Train
errors = nn.train(X, Y, epochs=200)

print(nn.forward(X[0]))
print(nn.forward(X[1]))
print(nn.forward(X[2]))
print(nn.forward(X[3]))

# Save
nn.save(model_name="xor")

# Visualization
fig = plt.figure("XOR Visualization", figsize=(12, 6))
fig.suptitle("XOR Problem")

# Error plot
ax = plt.subplot(2, 2, 1)
ax.plot(errors)
ax.set_xlabel("Epoch")
ax.set_ylabel("Error")

# 2D Plot
ax = plt.subplot(2, 2, 3)
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)
ax.set_xlabel("x1")
ax.set_ylabel("x2")

# Background color based on Z
ax.scatter(X[:, 0], X[:, 1], c=Y > 0.5, cmap="bwr")
xs = np.linspace(0, 1, 100)
ys = np.linspace(0, 1, 100)
Xg, Yg = np.meshgrid(xs, ys)
Z = np.zeros((100, 100))
for i in range(100):
    for j in range(100):
        Z[i, j] = nn.forward(np.array([Xg[i, j], Yg[i, j]])) > 0.5

ax.imshow(Z, extent=[0, 1, 0, 1], origin="lower", alpha=0.2)

# 3D Plot
ax = fig.add_subplot(1, 2, 2, projection="3d")
ax.set_title("3D")
ax.plot_surface(Xg, Yg, Z, cmap="viridis", edgecolor="none")
plt.show()