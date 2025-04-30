
import numpy as np
from scipy import linalg
from utils import seam_blend

class APAP:
    def __init__(self, canvas_shape, im1, im2, X1, X2, sigma=8.0, gamma=0.005):
        self.canvas_h, self.canvas_w = canvas_shape
        self.im1 = im1; self.im2 = im2
        self.X1 = X1; self.X2 = X2
        self.sigma = sigma; self.gamma = gamma

    def compute_vertex_H(self, vx, vy):
        pts1, pts2 = self.X1, self.X2
        N = pts1.shape[1]

        # spatial weights
        diffs = pts1[:2].T - np.array([vx, vy])
        d2 = np.sum(diffs**2, axis=1)
        w = np.exp(-d2 / (2 * self.sigma**2))

         # weights
        A = np.zeros((2*N, 9))
        for i in range(N):
            x, y, _ = pts1[:,i]
            u, v, _ = pts2[:,i]
            A[2*i]   = [ x*u,  y*u,  u, 0, 0, 0, -x*u, -y*u, -u ]
            A[2*i+1] = [ 0, 0, 0, x*v,  y*v,  v, -x*v, -y*v, -v ]
        sqrtw = np.repeat(np.sqrt(w), 2)
        Aw = A * sqrtw[:, None]

        # single vector decomposition
        _, _, Vt = linalg.svd(Aw)
        h = Vt[-1]; H = h.reshape(3,3)
        return H / H[2,2]

    def stitch(self, mesh_size=(10,10)):
        mh, mw = mesh_size
        ys = np.linspace(0, self.canvas_h, mh+1, dtype=int)
        xs = np.linspace(0, self.canvas_w, mw+1, dtype=int)
        Hvs = {}
        for i, xv in enumerate(xs):
            for j, yv in enumerate(ys):
                Hvs[(i,j)] = self.compute_vertex_H(xv, yv)
        warp = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)

         # finding the cell
        for y in range(self.canvas_h):
            for x in range(self.canvas_w):
                i = np.searchsorted(xs, x) - 1
                j = np.searchsorted(ys, y) - 1
                i = np.clip(i, 0, mw-1)
                j = np.clip(j, 0, mh-1)
                x0, x1 = xs[i], xs[i+1]
                y0, y1 = ys[j], ys[j+1]
                dx = (x - x0)/(x1-x0) if x1>x0 else 0
                dy = (y - y0)/(y1-y0) if y1>y0 else 0
                weights = {
                    (i, j):       (1-dx)*(1-dy),
                    (i+1, j):     dx*(1-dy),
                    (i, j+1):     (1-dx)*dy,
                    (i+1, j+1):   dx*dy
                }
                H_sum = np.zeros((3,3))
                w_sum = 0
                for (ii, jj), wgt in weights.items():
                    H_sum += wgt * Hvs[(ii, jj)]
                    w_sum += wgt
                Hloc = H_sum / w_sum
                p = Hloc @ np.array([x, y, 1])
                p /= p[2]
                xi, yi = int(p[0]), int(p[1])
                if 0 <= yi < self.im2.shape[0] and 0 <= xi < self.im2.shape[1]:
                    warp[y, x] = self.im2[yi, xi]
         # seaming begins blended : point to check for errors
        return seam_blend(self.im1, warp)

