import sys
import cv2
from Load_data import load_data
from Matcher import FeatureMatcher
from HM_ransac import HM_ransac
from mosaic_global import mosaic_global
from mosaic_local_ori import mosaic_local_ori
from apap import APAP
from comp_KR import comp_KR
from utils import calculate_RMSE
from constant import max_iteration, ransac_threshold, mesh_size, sigma, gamma
from compare_algorithms import compare_algorithms


def run_all(path1, path2, output_prefix):

    im1, im2, _, _ = load_data(path1, path2)

    # feature matching using few based on the sift , orb and flann
    matcher = FeatureMatcher('SIFT', 'FLANN', ratio=0.7)
    try:
        fp, tp = matcher.match(im1, im2)
    except RuntimeError as e:
        print(f"SIFT matching failed ({e}), falling back to ORB+BF")
        matcher = FeatureMatcher('ORB', 'BF', ratio=0.75)
        fp, tp = matcher.match(im1, im2)

    # ransac
    H, inliers, _ = HM_ransac(fp, tp, max_iteration, ransac_threshold)
    fp_in, tp_in = fp[:, inliers], tp[:, inliers]

    # global
    img_g = mosaic_global(im1, im2, H)
    cv2.imwrite(f"{output_prefix}_global.jpg", img_g)

    # apap
    canvas_h = max(im1.shape[0], im2.shape[0])
    canvas_w = im1.shape[1] + im2.shape[1]
    apap = APAP((canvas_h, canvas_w), im1, im2, fp_in, tp_in, sigma=sigma, gamma=gamma)
    img_a = apap.stitch(mesh_size)
    cv2.imwrite(f"{output_prefix}_apap.jpg", img_a)

    # rew_kr 
    Hk, X1k, X2k = comp_KR(im1, im2, fp_in, tp_in)
    img_k = mosaic_local_ori(im1, im2, Hk, X1k, X2k, mesh_size)
    cv2.imwrite(f"{output_prefix}_kr.jpg", img_k)

    # evaluation metrices
    compare_algorithms(path1, path2, save_results=True, output_prefix=output_prefix)


if __name__ == '__main__':
    img1_path=""
    img2_path=""
    try:
        run_all(img1_path, img2_path,"new")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
