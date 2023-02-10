import numpy as np
import matplotlib.pyplot as plt
import cv2


def load_image(path, res, grey=False):
    """
    Load an png image from a path
    """
    if grey:
        img = plt.imread(path, format="png")
        img = img.mean(axis=2)
    else:
        img = plt.imread(path, format="png")[:, :, 0]
    img = img / 255
    img = img.reshape(res, res)
    return img


def normalize_center_scale(img, res, pad=4):
    """
    1. Normalize image, setting its lowest value to 0 and its highest to 1
    2. Crop image to Bound Box
    3. Square image padding with 0s
    4. Pad image
    5. Resize image to res x res
    """

    # Check if image is blank
    if np.isclose(img.max(), 0):
        return np.zeros((res, res))

    # Normalize
    img = img - img.min()
    img = img / img.max()

    # Bound Box
    ys, xs = np.nonzero(img)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    # Crop image
    img_crop = img[y1 : y2 + 1, x1 : x2 + 1].copy()

    # Square image
    h, w = img_crop.shape
    if h > w:
        img_crop = np.pad(img_crop, ((0, 0), ((h - w) // 2, (h - w) // 2)), "constant")
    elif w > h:
        img_crop = np.pad(img_crop, (((w - h) // 2, (w - h) // 2), (0, 0)), "constant")

    # Pad image
    img_crop = np.pad(img_crop, pad, "constant")

    # Resize image using avg pooling
    img_final = cv2.resize(img_crop, (res, res), interpolation=cv2.INTER_AREA)

    return img_final


def accuracy(Y_true, Y_pred):
    """
    Calculate the accuracy of a classification model
    """
    Y_pred = np.argmax(Y_pred, axis=1)
    Y_true = np.argmax(Y_true, axis=1)
    return np.mean(Y_true == Y_pred)


def precision(Y_true, Y_pred):
    """
    Calculate the precision of a classification model
    """
    Y_pred = np.argmax(Y_pred, axis=1)
    Y_true = np.argmax(Y_true, axis=1)
    return np.sum(Y_true[Y_true == Y_pred] == 1) / np.sum(Y_pred == 1)


def recall(Y_true, Y_pred):
    """
    Calculate the recall of a classification model
    """
    Y_pred = np.argmax(Y_pred, axis=1)
    Y_true = np.argmax(Y_true, axis=1)
    return np.sum(Y_true[Y_true == Y_pred] == 1) / np.sum(Y_true == 1)


def f1_score(Y_true, Y_pred):
    """
    Calculate the f1_score of a classification model
    """
    Y_pred = np.argmax(Y_pred, axis=1)
    Y_true = np.argmax(Y_true, axis=1)
    p = precision(Y_true, Y_pred)
    r = recall(Y_true, Y_pred)
    return 2 * p * r / (p + r)