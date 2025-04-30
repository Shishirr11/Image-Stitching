import numpy as np
from scipy import linalg


def HM_ransac(fp, tp, max_iter=500, threshold=30.0):

    N = fp.shape[1]
    best_H = None
    best_inliers = np.array([], dtype=int)
    best_score = 0

    for _ in range(max_iter):
        # few samples max 4 for stable ouput until now
        idx = np.random.choice(N, 4, replace=False)
        A = []
        for i in idx:
            x, y, _ = fp[:, i]
            u, v, _ = tp[:, i]
            A.append([ x*u,  y*u,  u, 0, 0, 0, -x*u, -y*u, -u ])
            A.append([ 0, 0, 0, x*v,  y*v,  v, -x*v, -y*v, -v ])
        A = np.array(A)
        # single vector decomposition - for features
        _, _, Vt = linalg.svd(A)
        h = Vt[-1]
        H_candidate = h.reshape(3, 3)
        H_candidate = H_candidate / H_candidate[2, 2]

        # inliers
        proj = H_candidate @ fp
        proj[:2] /= proj[2]
        dists = np.linalg.norm(proj[:2] - tp[:2], axis=0)
        inliers = np.where(dists < threshold)[0]
        score = inliers.size

        if score > best_score:
            best_score = score
            best_inliers = inliers
            best_H = H_candidate

    if best_score >= 4:
        A = []
        for i in best_inliers:
            x, y, _ = fp[:, i]
            u, v, _ = tp[:, i]
            A.append([ x*u,  y*u,  u, 0, 0, 0, -x*u, -y*u, -u ])
            A.append([ 0, 0, 0, x*v,  y*v,  v, -x*v, -y*v, -v ])
        A = np.array(A)
        _, _, Vt = linalg.svd(A)
        h = Vt[-1]
        H_refined = h.reshape(3, 3)
        best_H = H_refined / H_refined[2, 2]

    return best_H, best_inliers, best_score