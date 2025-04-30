import numpy as np
from HM_ransac import HM_ransac


def comp_KR(im1, im2, X1, X2, max_iter=500, thresh=30.0):


    #ransac and noramlize
    H, inliers, _ = HM_ransac(X1, X2, max_iter, thresh)
    H = H / H[2, 2]
    print(f"RANSAC inliers: {len(inliers)} / {X1.shape[1]}")

    
    X1_ok = X1[:, inliers]
    X2_ok = X2[:, inliers]

    # reprojection error on inliers
    proj_in = H @ X1_ok
    proj_in[:2] /= proj_in[2]
    errs_in = np.linalg.norm(proj_in[:2] - X2_ok[:2], axis=0)
    rmse_in = np.sqrt(np.mean(errs_in**2))
    print(f"Inlier reprojection RMSE: {rmse_in:.3f} pixels")

    proj_all = H @ X1
    proj_all[:2] /= proj_all[2]
    errs_all = np.linalg.norm(proj_all[:2] - X2[:2], axis=0)
    rmse_all = np.sqrt(np.mean(errs_all**2))
    print(f"All-point reprojection RMSE: {rmse_all:.3f} pixels")

    cond_num = np.linalg.cond(H)
    print(f"Homography condition number: {cond_num:.1f}")

    return H, X1_ok, X2_ok
