import numpy as np
from scipy import linalg
from scipy.ndimage import map_coordinates
from utils import seam_blend

def mosaic_global(im1, im2, H):
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]

    corners = np.array([[0, w2-1, w2-1, 0],
                        [0, 0, h2-1, h2-1],
                        [1, 1, 1, 1]])

    # mapingg the imageee corners
    warped = linalg.inv(H) @ corners
    warped[:2] /= warped[2]

    x_min = int(np.floor(min(0, warped[0].min())))
    x_max = int(np.ceil(max(w1-1, warped[0].max())))
    y_min = int(np.floor(min(0, warped[1].min())))
    y_max = int(np.ceil(max(h1-1, warped[1].max())))
    x_min = max(x_min, -w2)
    x_max = min(x_max, w1 + w2)
    y_min = max(y_min, -h2)
    y_max = min(y_max, h1 + h2)

    xs = np.arange(x_min, x_max + 1, dtype=int)
    ys = np.arange(y_min, y_max + 1, dtype=int)

    # THIS is the culprit for coordinate mismatches
    U, V = np.meshgrid(xs, ys)
    U = U.astype(int)
    V = V.astype(int)

    # prepare warped images
    out1 = np.zeros((len(ys), len(xs), 3), dtype=np.uint8)
    out2 = np.zeros_like(out1)
    m1 = (0 <= V) & (V < h1) & (0 <= U) & (U < w1)
    out1[m1] = im1[V[m1], U[m1]]

    pts = np.vstack((U.ravel(), V.ravel(), np.ones_like(U.ravel())))
    coords = H @ pts
    coords[:2] /= coords[2]
    x2 = coords[0].reshape(V.shape); y2 = coords[1].reshape(V.shape)
    for c in range(3):
        out2[..., c] = map_coordinates(im2[..., c], [y2, x2], order=1, mode='constant', cval=0)

    # blend via seam cut
    return seam_blend(out1, out2)

