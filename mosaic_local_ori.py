import numpy as np
from scipy import linalg
from scipy.ndimage import map_coordinates
from utils import uniform_blend
from utils import seam_blend

def mosaic_local_ori(im1, im2, H, X1, X2, mesh_size=(10, 10)):
    h1, w1 = im1.shape[:2]
    ys = np.linspace(0, h1, mesh_size[1] + 1, dtype=int)
    xs = np.linspace(0, w1, mesh_size[0] + 1, dtype=int)

    # doing the per - cell homographies
    H_cells = {}
    for i in range(mesh_size[0]):
        for j in range(mesh_size[1]):
            A = []
            for p, q in zip(X1.T, X2.T):
                x, y, _ = p
                u, v, _ = q
                A.append([ x*u,  y*u,  u, 0, 0, 0, -x*u, -y*u, -u ])
                A.append([ 0, 0, 0, x*v,  y*v,  v, -x*v, -y*v, -v ])
            _, _, Vt = linalg.svd(np.array(A))
            H_cells[(i, j)] = Vt[-1].reshape(3, 3)

    # warp into common canvas
    warp2 = np.zeros_like(im1)
    for y in range(h1):
        for x in range(w1):
            ci = min(x // (w1 // mesh_size[0]), mesh_size[0] - 1)
            cj = min(y // (h1 // mesh_size[1]), mesh_size[1] - 1)
            Hc = H_cells[(ci, cj)]
            pt = Hc @ np.array([x, y, 1])
            pt /= pt[2]
            xi, yi = int(pt[0]), int(pt[1])
            if 0 <= yi < im2.shape[0] and 0 <= xi < im2.shape[1]:
                warp2[y, x] = im2[yi, xi]

    return seam_blend(im1, warp2)

