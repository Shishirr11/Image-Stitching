import cv2
import numpy as np
from Seam import find_seam

def calculate_RMSE(img1, img2):
    mask = (np.sum(img1, axis=-1) > 0) & (np.sum(img2, axis=-1) > 0)
    if not np.any(mask):
        return float('nan')
    diff = (img1.astype(np.float64) - img2.astype(np.float64))**2
    return float(np.sqrt(np.mean(diff[mask])))

def uniform_blend(im1, im2):
    h = max(im1.shape[0], im2.shape[0])
    w = max(im1.shape[1], im2.shape[1])
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    mask1 = (np.pad(im1, ((0,h-im1.shape[0]),(0,w-im1.shape[1]),(0,0))) > 0).any(axis=-1)
    mask2 = (np.pad(im2, ((0,h-im2.shape[0]),(0,w-im2.shape[1]),(0,0))) > 0).any(axis=-1)
    canvas[mask1] = np.pad(im1, ((0,h-im1.shape[0]),(0,w-im1.shape[1]),(0,0)))[mask1]
    canvas[mask2] = np.pad(im2, ((0,h-im2.shape[0]),(0,w-im2.shape[1]),(0,0)))[mask2]
    overlap = mask1 & mask2
    for c in range(3):
        ch1 = np.pad(im1[...,c], ((0,h-im1.shape[0]),(0,w-im1.shape[1])))
        ch2 = np.pad(im2[...,c], ((0,h-im2.shape[0]),(0,w-im2.shape[1])))
        canvas[...,c][overlap] = ((ch1[overlap].astype(np.float64) + ch2[overlap].astype(np.float64))/2).astype(np.uint8)
    return canvas

def seam_blend(im1, im2):
    h, w = im1.shape[:2]
    E = find_seam(im1, im2)
    M = E.astype(np.float64)
    back = np.zeros_like(M, dtype=int)
    for i in range(1, h):
        for j in range(w):
            j0 = max(j-1, 0)
            j1 = min(j+1, w-1)
            idx = np.argmin(M[i-1, j0:j1+1])
            back[i, j] = idx + j0
            M[i, j] += M[i-1, back[i, j]]
    seam = np.zeros(h, dtype=int)
    seam[-1] = np.argmin(M[-1])
    for i in range(h-2, -1, -1):
        seam[i] = back[i+1, seam[i+1]]
    out = np.zeros_like(im1)
    for i in range(h):
        out[i, :seam[i]] = im1[i, :seam[i]]
        out[i, seam[i]:] = im2[i, seam[i]:]
    return out