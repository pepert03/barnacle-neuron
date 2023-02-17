import sys
import os
from sklearn.model_selection import train_test_split

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


from package.neunet import *

# Create Dataset
X = np.random.rand(400, 2)
Y = np.array([])
for x, y in X:
    if (x - 0.5) ** 2 + (y - 0.5) ** 2 < 0.2**2:
        Y = np.append(Y, [0, 1, 0])
    elif x < 0.5:
        Y = np.append(Y, [1, 0, 0])
    else:
        Y = np.append(Y, [0, 0, 1])
Y = Y.reshape(400, 3)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# activate_debug_mode()

# Neural Network Structure
layers = [
    Dense(2, 10),
    Sigmoid(10),
    Dense(10, 3),
    Softmax(3),
    CrossEntropy(3),
]

# Create Neural Network
nn = NeuNet(layers)

# Compile
nn.compile(learning_rate=0.1, metrics=["accuracy", "recall", "precision"])

# Train
errors = nn.fit(X_train, Y_train, epochs=250, verbose=True)

# Test
nn.evaluate(X_test, Y_test)

# Save model
print(nn.layers)
nn.save(model_path="models/circular3class")

# Visualizations
fig = plt.figure("Circular Visualization", figsize=(12, 6))

fig.suptitle("Visualization of Circular Clasification")

# Error plot
ax = plt.subplot(2, 2, 1)
ax.set_title("Error")
ax.plot(errors)

# 2d plot of data
ax = plt.subplot(2, 2, 3)
ax.set_title("2D")
ax.scatter(X[:, 0], X[:, 1], c=np.argmax(Y, axis=1))

# Background color based on Z
xs = np.linspace(0, 1, 100)
ys = np.linspace(0, 1, 100)
Xg, Yg = np.meshgrid(xs, ys)
Z = np.zeros((100, 100))
for i in range(100):
    for j in range(100):
        Z[i, j] = np.argmax(nn.predict(np.array([Xg[i, j], Yg[i, j]])))

ax.imshow(Z, extent=[0, 1, 0, 1], origin="lower", alpha=0.2)

# 3d plot of Z in the center of the second row
ax = fig.add_subplot(1, 2, 2, projection="3d")
ax.set_title("3D")
ax.plot_surface(Xg, Yg, Z, cmap="viridis", edgecolor="none")
plt.show()
