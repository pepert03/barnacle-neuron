# Convolution pruebas
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

from package.neunet import *

# numpy warnings to errors
np.seterr(all="raise")

# Load data
X = np.load("data/X.npy")

# Hacer convolucion 3 tipos
kernel1 = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
kernel2 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
kernel3 = np.array([[1, 1, 1], [1, 0, -1], [1, -1, -1]])

# Hacer convoluciones a la primera imagen de X y mostrarlas
img = X[random.randint(0, len(X))].reshape(28, 28)
img1 = signal.convolve2d(img, kernel1, mode="same")
img2 = signal.convolve2d(img, kernel2, mode="same")
img3 = signal.convolve2d(img, kernel3, mode="same")

# Mostrar imagenes
fig = plt.figure("Convolution Visualization", figsize=(12, 6))
fig.suptitle("Visualization of Convolution")

ax = plt.subplot(2, 2, 1)
ax.set_title("Original")
ax.imshow(img, cmap="gray")

ax = plt.subplot(2, 2, 2)
ax.set_title("Kernel 1")
ax.imshow(img1, cmap="gray")

ax = plt.subplot(2, 2, 3)
ax.set_title("Kernel 2")
ax.imshow(img2, cmap="gray")

ax = plt.subplot(2, 2, 4)
ax.set_title("Kernel 3")
ax.imshow(img3, cmap="gray")

plt.show()