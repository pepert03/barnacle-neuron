import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d

# Nupy warnings to errors
np.seterr(all="raise")


def load_image(path, grey=False):
    """
    Load an png image from a path
    """
    if grey:
        img = plt.imread(path)
        img = img.mean(axis=2)
    else:
        img = plt.imread(path)
    img = img / 255
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
    if len(Y_pred.shape) == 1:
        Y_pred_ = np.zeros((Y_pred.shape[0], 2))
        Y_pred_[:, 0] = Y_pred
        Y_pred_[:, 1] = 1 - Y_pred
        Y_true = np.zeros((Y_true.shape[0], 2))
        Y_true[:, 0] = Y_true
        Y_true[:, 1] = 1 - Y_true

    Y_pred = np.argmax(Y_pred, axis=1)
    Y_true = np.argmax(Y_true, axis=1)

    return np.mean(Y_true == Y_pred)


def precision(Y_true, Y_pred):
    """
    Calculate the precision of a classification model
    """
    if len(Y_pred.shape) == 1:
        Y_pred_ = np.zeros((Y_pred.shape[0], 2))
        Y_pred_[:, 0] = Y_pred
        Y_pred_[:, 1] = 1 - Y_pred
        Y_true = np.zeros((Y_true.shape[0], 2))
        Y_true[:, 0] = Y_true
        Y_true[:, 1] = 1 - Y_true
    n_classes = Y_pred.shape[1]
    Y_pred = np.argmax(Y_pred, axis=1)
    Y_true = np.argmax(Y_true, axis=1)
    mean_precision = 0
    for label in range(n_classes):
        tp = np.sum(Y_true[Y_true == Y_pred] == label)
        fp = np.sum(Y_true[Y_true != Y_pred] == label)
        mean_precision += tp / (tp + fp)

    return mean_precision / n_classes


def recall(Y_true, Y_pred):
    """
    Calculate the recall of a classification model
    """
    if len(Y_pred.shape) == 1:
        Y_pred_ = np.zeros((Y_pred.shape[0], 2))
        Y_pred_[:, 0] = Y_pred
        Y_pred_[:, 1] = 1 - Y_pred
        Y_true = np.zeros((Y_true.shape[0], 2))
        Y_true[:, 0] = Y_true
        Y_true[:, 1] = 1 - Y_true
    n_classes = Y_pred.shape[1]
    Y_pred = np.argmax(Y_pred, axis=1)
    Y_true = np.argmax(Y_true, axis=1)
    mean_recall = 0
    for label in range(n_classes):
        tp = np.sum(Y_true[Y_true == Y_pred] == label)
        fn = np.sum(Y_true[Y_true != Y_pred] != label)
        mean_recall += tp / (tp + fn)
    return mean_recall / n_classes


def f1_score(Y_true, Y_pred):
    """
    Calculate the f1_score of a classification model
    """
    if len(Y_pred.shape) == 1:
        Y_pred_ = np.zeros((Y_pred.shape[0], 2))
        Y_pred_[:, 0] = Y_pred
        Y_pred_[:, 1] = 1 - Y_pred
        Y_true = np.zeros((Y_true.shape[0], 2))
        Y_true[:, 0] = Y_true
        Y_true[:, 1] = 1 - Y_true
    Y_pred = np.argmax(Y_pred, axis=1)
    Y_true = np.argmax(Y_true, axis=1)
    p = precision(Y_true, Y_pred)
    r = recall(Y_true, Y_pred)
    return 2 * p * r / (p + r)


def convolve3d(image, kernel):
    """Convolves a 3d image with a 3d filter:
    Image: (H1, W1, D1) (D1>=D2)
    Kernel: (H2, W2, D2)
    Returns: (H1-H2+1, H1-H2+1, D1-D2+1)
    """
    output = np.zeros(
        (
            image.shape[0] - kernel.shape[0] + 1,
            image.shape[1] - kernel.shape[1] + 1,
            image.shape[2] - kernel.shape[2] + 1,
        )
    )

    for d1 in range(image.shape[2] - kernel.shape[2] + 1):
        for d2 in range(kernel.shape[2]):
            output[:, :, d1] += convolve2d(
                image[:, :, d1 + d2], kernel[:, :, d2], mode="valid"
            )

    return output


def convolve4d(image, kernels):
    """Convolves a 3d image with a 4d filter:
    Image: (H1, W1, D1)
    Filter: (H2, W2, D2, N)
    Returns: (H1-H2+1, W1-W2+1,(D1-D2+1)*N)
    """

    D = image.shape[2] - kernels.shape[2] + 1

    output = np.zeros(
        (
            image.shape[0] - kernels.shape[0] + 1,
            image.shape[1] - kernels.shape[1] + 1,
            kernels.shape[3] * D,
        )
    )

    for i in range(0, output.shape[2], D):
        output[:, :, i : i + D] = convolve3d(image, kernels[:, :, :, i // D])

    return output


# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# im = load_image(os.path.join(parent_dir, "assets", "logo.jpg"))
# Show image
# plt.imshow(im)
# plt.show()

# convolve image with a 3x3x3 filter
# print(im[:5,:5,:])
# filter_ = np.random.rand(3, 3, 1)
# print(filter_)

# im_conv = convolve3d(im, filter_)

# print(im_conv.shape)

# plt.imshow(im_conv)
# plt.show()

# 4d filter
# filter_ = np.random.randn(3, 3, 1, 6)

# output = convolve4d(im,filter_)

# print(output.shape)

# for i in range(6):
#     plt.subplot(2, 3, i + 1)
#     plt.imshow(output[:, :, i:i+3], cmap="gray")

# plt.show()
