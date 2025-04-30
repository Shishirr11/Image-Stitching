import cv2
import numpy as np


def calculate_RMSE(img1, img2):

    mask = (np.sum(img1, axis=-1) > 0) & (np.sum(img2, axis=-1) > 0)
    if not np.any(mask):
        return float('nan')
    diff = (img1.astype(np.float64) - img2.astype(np.float64))**2
    mse = np.mean(diff[mask])
    return float(np.sqrt(mse))


def uniform_blend(im1, im2):

    h = max(im1.shape[0], im2.shape[0])
    w = max(im1.shape[1], im2.shape[1])
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    mask1 = np.zeros((h, w), bool)
    mask2 = np.zeros((h, w), bool)

    mask1[:im1.shape[0], :im1.shape[1]] = np.sum(im1, axis=-1) > 0
    mask2[:im2.shape[0], :im2.shape[1]] = np.sum(im2, axis=-1) > 0

    canvas[:im1.shape[0], :im1.shape[1]][mask1[:im1.shape[0], :im1.shape[1]]] = \
        im1[mask1[:im1.shape[0], :im1.shape[1]]]
    canvas[:im2.shape[0], :im2.shape[1]][mask2[:im2.shape[0], :im2.shape[1]]] = \
        im2[mask2[:im2.shape[0], :im2.shape[1]]]

    overlap = mask1 & mask2
    for c in range(3):
        ch1 = np.zeros((h, w), dtype=np.uint8)
        ch2 = np.zeros((h, w), dtype=np.uint8)
        ch1[:im1.shape[0], :im1.shape[1]] = im1[..., c]
        ch2[:im2.shape[0], :im2.shape[1]] = im2[..., c]
        avg = ((ch1.astype(np.float64) + ch2.astype(np.float64)) / 2).astype(np.uint8)
        canvas[..., c][overlap] = avg[overlap]

    return canvas


def cv_draw_matches(img1, kp1, img2, kp2, matches, flags=cv2.DrawMatchesFlags_DEFAULT):
    return cv2.drawMatches(img1, [], img2, [], matches, None, flags=flags)