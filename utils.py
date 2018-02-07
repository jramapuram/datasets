import cv2
import numpy as np
from PIL import Image

def resize_lambda(img, size=(64, 64)):
    if not isinstance(img, (np.float32, np.float64)):
        img = np.asarray(img)

    if not isinstance(size, tuple):
        size = tuple(size)

    return cv2.resize(img, size)

# def bw_2_rgb_lambda(img):
#     if not isinstance(img, (np.float32, np.float64)):
#         img = np.asarray(img)

#     return cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

def bw_2_rgb_lambda(img):
    if img.mode == "RGB":
        return img

    # if not isinstance(img, (np.float32, np.float64)):
    #     img = np.asarray(img)

    # img = np.expand_dims(img, -1)
    # img = np.concatenate((img, img, img), axis=-1)
    # print("img =", img, " | dtype = ", img.dtype)
    # return Image.fromarray(img.astype('uint8'))

    return img.convert(mode="RGB")
    #return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)#.transpose(2, 1, 0)
