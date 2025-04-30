import cv2
import numpy as np

class FeatureMatcher:
    def __init__(self, detector_type='SIFT', matcher_type='FLANN', ratio=0.7):
        self.ratio = ratio
        detector_type = detector_type.upper()
        if detector_type == 'SIFT':
            self.detector = cv2.SIFT_create()
        elif detector_type == 'ORB':
            self.detector = cv2.ORB_create()
        else:
            raise ValueError(f" wrong detector type: {detector_type}")

        matcher_type = matcher_type.upper()
        if matcher_type == 'FLANN':
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        elif matcher_type == 'BF':
            norm = cv2.NORM_L2 if detector_type == 'SIFT' else cv2.NORM_HAMMING
            self.matcher = cv2.BFMatcher(norm)
        else:
            raise ValueError(f"wrong matcher type: {matcher_type}")

    def match(self, img1, img2):
        gray1 = img1 if len(img1.shape)==2 else cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = img2 if len(img2.shape)==2 else cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        kp1, des1 = self.detector.detectAndCompute(gray1, None)
        kp2, des2 = self.detector.detectAndCompute(gray2, None)
        if des1 is None or des2 is None:
            raise RuntimeError(" no descriptors for images")

        raw_matches = self.matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in raw_matches:
            if m.distance < self.ratio * n.distance:
                good_matches.append(m)

        if len(good_matches) < 4:
            raise RuntimeError("neeed more features")

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        fp = np.vstack((pts1.T, np.ones((1, pts1.shape[0]))))
        tp = np.vstack((pts2.T, np.ones((1, pts2.shape[0]))))
        return fp, tp
