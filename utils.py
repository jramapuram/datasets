import cv2
import numpy as np

def resize_lambda(img, size=(64, 64)):
    if not isinstance(img, (np.float32, np.float64)):
        img = np.asarray(img)

    if not isinstance(size, tuple):
        size = tuple(size)

    return cv2.resize(img, size)

def bw_2_rgb_lambda(img):
    if not isinstance(img, (np.float32, np.float64)):
        img = np.asarray(img)

    return cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
