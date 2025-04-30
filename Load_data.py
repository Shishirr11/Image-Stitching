import cv2
def load_data(path1, path2, flag=cv2.IMREAD_COLOR):
    im1 = cv2.imread(path1, flag)
    im2 = cv2.imread(path2, flag)
    missing = []
    if im1 is None:
        missing.append(path1)
    if im2 is None:
        missing.append(path2)
    if missing:
        raise FileNotFoundError("no image idiot")

    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    return im1, im2, gray1, gray2
