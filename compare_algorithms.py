import cv2
import time
import numpy as np
from Matcher import FeatureMatcher
from HM_ransac import HM_ransac
from mosaic_global import mosaic_global
from apap import APAP
from comp_KR import comp_KR
from mosaic_local_ori import mosaic_local_ori
from utils import calculate_RMSE
from constant import max_iteration, ransac_threshold, mesh_size, sigma, gamma

def compare_algorithms(path1, path2, save_results=True, output_prefix='result'):
    im1 = cv2.imread(path1)
    im2 = cv2.imread(path2)
    if im1 is None or im2 is None:
        raise FileNotFoundError(f"Cannot load images: {path1}, {path2}")

    matcher = FeatureMatcher('SIFT', 'FLANN', ratio=0.7)
    try:
        fp, tp = matcher.match(im1, im2)
    except RuntimeError as e:
        print(f"SIFT matching failed ({e}), falling back to ORB+BF")
        matcher = FeatureMatcher('ORB', 'BF', ratio=0.75)
        fp, tp = matcher.match(im1, im2)

    H, inliers, _ = HM_ransac(fp, tp, max_iteration, ransac_threshold)
    fp_in = fp[:, inliers]
    tp_in = tp[:, inliers]

    results = {}
    outputs = {}

    # Global mosaic
    start = time.time()
    img_g = mosaic_global(im1, im2, H)
    t_g = time.time() - start
    results['Global'] = {'time': t_g, 'inliers': len(inliers)}
    outputs['Global'] = img_g
    if save_results:
        cv2.imwrite(f"{output_prefix}_global.jpg", img_g)
        print("Saved global mosaic")

    # APAP mosaic
    start = time.time()
    canvas_h = max(im1.shape[0], im2.shape[0])
    canvas_w = im1.shape[1] + im2.shape[1]
    apap = APAP((canvas_h, canvas_w), im1, im2, fp_in, tp_in, sigma=sigma, gamma=gamma)
    img_a = apap.stitch(mesh_size)
    t_a = time.time() - start
    results['APAP'] = {'time': t_a}
    outputs['APAP'] = img_a
    if save_results:
        cv2.imwrite(f"{output_prefix}_apap.jpg", img_a)
        print("Saved APAP mosaic")

    # KR (REW) mosaic
    start = time.time()
    Hk, X1k, X2k = comp_KR(im1, im2, fp_in, tp_in)
    img_k = mosaic_local_ori(im1, im2, Hk, X1k, X2k, mesh_size)
    t_k = time.time() - start
    results['KR'] = {'time': t_k}
    outputs['KR'] = img_k
    if save_results:
        cv2.imwrite(f"{output_prefix}_kr.jpg", img_k)
        print("Saved KR mosaic")

    # Pairwise RMSE
    print("\nPairwise RMSE:")
    keys = list(outputs.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            k1, k2 = keys[i], keys[j]
            rmse = calculate_RMSE(outputs[k1], outputs[k2])
            print(f"{k1} vs {k2}: {rmse:.3f}")

    print("\nComparsions:")
    for name, data in results.items():
        line = f"{name}: time={data['time']:.3f}s"
        if 'inliers' in data:
            line += f", inliers={data['inliers']}"
        print(line)


if __name__ == '__main__':
    img1_path=""
    img2_path=""
    compare_algorithms(img1_path, img2_path, save_results=True, output_prefix="through_CA_file")


