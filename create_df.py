import pandas as pd
import numpy as np
import os
import random as rd
import matplotlib.pyplot as plt

N_DECIMALS = 3


def png_to_numpy_flat(path: str):
    img = plt.imread(path)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    assert np.allclose(r, g) and np.allclose(r, b), "Image is not grayscale"
    return r.flatten()


def label_to_one_hot(label: int):
    one_hot = np.zeros(10)
    one_hot[label] = 1
    return one_hot


def build_csv(folder: str = "data"):
    X = np.array([])
    Y = np.array([])
    for i in range(10):
        for file in os.listdir(f"./{folder}/{i}"):
            path = f"./{folder}/{i}/{file}"
            im = png_to_numpy_flat(path)
            X = np.append(X, im)
            Y = np.append(Y, label_to_one_hot(i))

    # To csv
    X_df = pd.DataFrame(X.reshape(-1, 28 * 28).round(N_DECIMALS))
    Y_df = pd.DataFrame(Y.reshape(-1, 10).round(N_DECIMALS))

    X_df.to_csv(f"./{folder}/X.csv", index=False)
    Y_df.to_csv(f"./{folder}/Y.csv", index=False)


def build_npy(folder: str = "data"):
    X = np.array([])
    Y = np.array([])
    for i in range(10):
        for file in os.listdir(f"./{folder}/{i}"):
            path = f"./{folder}/{i}/{file}"
            im = png_to_numpy_flat(path)
            X = np.append(X, im)
            Y = np.append(Y, label_to_one_hot(i))

    # To csv
    X_df = X.reshape(-1, 28 * 28).round(N_DECIMALS)
    Y_df = Y.reshape(-1, 10).round(N_DECIMALS)

    np.save(f"./{folder}/X.npy", X_df)
    np.save(f"./{folder}/Y.npy", Y_df)


if __name__ == "__main__":
    # build_csv(folder="data")
    # build_npy(folder="data")
    X = np.load("data/X.npy")
    Y = np.load("data/Y.npy")
