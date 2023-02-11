import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


import numpy as np
from sklearn.model_selection import train_test_split
from package.neunet import *


# Load data
X = np.load("data/X.npy")
Y = np.load("data/Y.npy")

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Create layers
layers = [
    Dense(784, 10),
    Softmax(10),
    CrossEntropy(10),
]

# Create model
nn = NeuNet(layers)

# Compile
nn.compile(learning_rate=0.01, metrics=["accuracy", "recall", "precision"])

# Train
errors = nn.fit(X_train, Y_train, 30, verbose=True)

# Test
nn.evaluate(X_test, Y_test)

# Save model
nn.save(model_path="models/mnist")

# Visualizations
fig = plt.figure("MNIST Visualization", figsize=(12, 6))

fig.suptitle("Visualization of MNIST Clasification")

# Error plot
ax = plt.subplot(1, 1, 1)
ax.set_title("Error")
ax.plot(errors)

plt.show()
